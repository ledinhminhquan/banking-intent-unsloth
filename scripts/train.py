"""
train.py
--------
Fine-tune a pre-trained LLM for banking intent classification using Unsloth
(4-bit LoRA) and the HuggingFace SFTTrainer.
"""

import os
import json
import argparse
import time
from difflib import get_close_matches

import yaml
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Prompt template (Alpaca-style) – must be identical at training & inference
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n"
    "Classify the intent of the following banking customer message. "
    "Reply with only the intent label, nothing else.\n\n"
    "### Input:\n{text}\n\n"
    "### Response:\n{label}"
)

PROMPT_TEMPLATE_INFERENCE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n"
    "Classify the intent of the following banking customer message. "
    "Reply with only the intent label, nothing else.\n\n"
    "### Input:\n{text}\n\n"
    "### Response:\n"
)


def format_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """Convert a DataFrame into a HuggingFace Dataset with formatted prompts."""
    eos = tokenizer.eos_token
    records = []
    for _, row in df.iterrows():
        prompt = PROMPT_TEMPLATE.format(text=row["text"], label=row["label_text"])
        records.append({"formatted_text": prompt + eos})
    return Dataset.from_pandas(pd.DataFrame(records))


def evaluate_model(model, tokenizer, test_df, id2label, gen_cfg):
    """Run inference on the test set and return accuracy + classification report."""
    FastLanguageModel.for_inference(model)

    all_labels = list(set(id2label.values()))
    y_true, y_pred = [], []

    print(f"\nEvaluating on {len(test_df)} test samples …")
    start = time.time()

    for idx, row in test_df.iterrows():
        prompt = PROMPT_TEMPLATE_INFERENCE.format(text=row["text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=gen_cfg.get("max_new_tokens", 64),
                temperature=gen_cfg.get("temperature", 0.0),
                do_sample=False,
                use_cache=True,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        predicted = response.split("\n")[0].strip()

        if predicted not in all_labels:
            matches = get_close_matches(predicted, all_labels, n=1, cutoff=0.4)
            predicted = matches[0] if matches else predicted

        y_true.append(row["label_text"])
        y_pred.append(predicted)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  [{idx+1}/{len(test_df)}]  elapsed {elapsed:.1f}s")

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    elapsed = time.time() - start

    print(f"\nTest Accuracy : {acc:.4f}  ({sum(a==b for a,b in zip(y_true,y_pred))}/{len(y_true)})")
    print(f"Evaluation time: {elapsed:.1f}s")
    print(f"\nClassification Report:\n{report}")

    return {
        "accuracy": acc,
        "num_correct": sum(a == b for a, b in zip(y_true, y_pred)),
        "num_total": len(y_true),
        "classification_report": report,
        "predictions": [
            {"text": t, "true": yt, "pred": yp}
            for t, yt, yp in zip(test_df["text"].tolist(), y_true, y_pred)
        ],
    }


def main(config_path: str):
    # ---- Load config ----
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    save_cfg = cfg["save"]
    eval_cfg = cfg.get("evaluate", {})

    # ---- Load preprocessed data ----
    train_df = pd.read_csv(data_cfg["train_file"])
    test_df = pd.read_csv(data_cfg["test_file"])

    with open(data_cfg["label_map_file"], encoding="utf-8") as f:
        label_data = json.load(f)
    id2label = label_data["id2label"]

    print(f"Training samples : {len(train_df)}")
    print(f"Test samples     : {len(test_df)}")
    print(f"Num labels       : {label_data['num_labels']}")

    # ---- Load model ----
    print(f"\nLoading model: {model_cfg['name']} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=model_cfg.get("dtype"),
    )

    # ---- Apply LoRA adapters ----
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg.get("random_state", 3407),
    )

    # ---- Format training dataset ----
    train_dataset = format_dataset(train_df, tokenizer)
    print(f"Formatted training dataset: {len(train_dataset)} examples")

    # ---- Training arguments ----
    use_bf16 = is_bfloat16_supported()
    training_args = TrainingArguments(
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        max_steps=train_cfg["max_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=train_cfg["logging_steps"],
        optim=train_cfg["optim"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        seed=train_cfg["seed"],
        output_dir=train_cfg["output_dir"],
        save_strategy=train_cfg["save_strategy"],
        report_to=train_cfg.get("report_to", "none"),
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="formatted_text",
        max_seq_length=model_cfg["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # ---- Train ----
    print("\n========== Starting training ==========")
    train_start = time.time()
    trainer_stats = trainer.train()
    train_elapsed = time.time() - train_start
    print(f"Training completed in {train_elapsed:.1f}s")
    print(f"Training loss: {trainer_stats.training_loss:.4f}")

    # ---- Save model ----
    save_dir = save_cfg["model_dir"]
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nModel saved to: {save_dir}/")

    # ---- Evaluate ----
    if eval_cfg.get("run_eval_after_training", True):
        gen_cfg = {
            "max_new_tokens": eval_cfg.get("max_new_tokens", 64),
            "temperature": eval_cfg.get("temperature", 0.0),
        }
        results = evaluate_model(model, tokenizer, test_df, id2label, gen_cfg)

        results_file = eval_cfg.get("results_file", "evaluation_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": results["accuracy"],
                    "num_correct": results["num_correct"],
                    "num_total": results["num_total"],
                    "classification_report": results["classification_report"],
                    "training_loss": trainer_stats.training_loss,
                    "training_time_seconds": train_elapsed,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nEvaluation results saved to: {results_file}")

    print("\n========== Done ==========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune model with Unsloth")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()
    main(args.config)
