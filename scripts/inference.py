"""
inference.py
------------
Standalone inference module for banking intent classification.
Loads a fine-tuned Unsloth/LoRA checkpoint and predicts the intent of a
single text input.

Usage examples:
    # Single message
    python scripts/inference.py --config configs/inference.yaml \
        --message "I need to activate my new card"

    # Evaluate full test set
    python scripts/inference.py --config configs/inference.yaml --evaluate

    # Interactive mode
    python scripts/inference.py --config configs/inference.yaml --interactive
"""

import os
import json
import argparse
import time
import warnings
from difflib import get_close_matches

import yaml
import torch
import pandas as pd
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings(
    "ignore",
    message=r"Both `max_new_tokens` .* `max_length`.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)

# Must match the template used during training (see scripts/train.py)
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n"
    "Classify the intent of the following banking customer message. "
    "Reply with only the intent label, nothing else.\n\n"
    "### Input:\n{text}\n\n"
    "### Response:\n"
)


class IntentClassification:
    """
    Banking intent classifier that wraps a fine-tuned Unsloth model.

    Parameters
    ----------
    model_path : str
        Path to a YAML configuration file that contains at least the path to
        the saved model checkpoint, label map, and generation settings.
    """

    def __init__(self, model_path: str):
        with open(model_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_cfg = config["model"]
        data_cfg = config["data"]
        self.gen_cfg = config.get("generation", {})

        checkpoint = model_cfg["checkpoint_path"]
        print(f"Loading model from: {checkpoint} ...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=model_cfg.get("max_seq_length", 512),
            load_in_4bit=model_cfg.get("load_in_4bit", True),
            dtype=model_cfg.get("dtype"),
        )
        FastLanguageModel.for_inference(self.model)

        with open(data_cfg["label_map_file"], encoding="utf-8") as f:
            label_data = json.load(f)

        self.id2label = label_data["id2label"]
        self.label2id = label_data["label2id"]
        self.all_labels = list(self.label2id.keys())
        print(f"Loaded {len(self.all_labels)} intent labels.")

    def __call__(self, message: str) -> str:
        """
        Predict the intent label of a banking customer message.

        Parameters
        ----------
        message : str
            A single banking customer query.

        Returns
        -------
        str
            The predicted intent label.
        """
        prompt = PROMPT_TEMPLATE.format(text=message.strip().lower())
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_cfg.get("max_new_tokens", 64),
                temperature=self.gen_cfg.get("temperature", 0.0),
                do_sample=self.gen_cfg.get("do_sample", False),
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        predicted_label = response.split("\n")[0].strip()

        if predicted_label not in self.all_labels:
            matches = get_close_matches(
                predicted_label, self.all_labels, n=1, cutoff=0.4
            )
            if matches:
                predicted_label = matches[0]

        return predicted_label


def evaluate(classifier: IntentClassification, test_file: str):
    """Evaluate the classifier on the full test set."""
    test_df = pd.read_csv(test_file)
    y_true, y_pred = [], []

    print(f"\nRunning evaluation on {len(test_df)} test samples ...\n")
    start = time.time()

    for idx, row in test_df.iterrows():
        pred = classifier(row["text"])
        y_true.append(row["label_text"])
        y_pred.append(pred)

        if (idx + 1) % 50 == 0:
            acc_so_far = accuracy_score(y_true, y_pred)
            elapsed = time.time() - start
            print(f"  [{idx+1}/{len(test_df)}]  acc={acc_so_far:.4f}  elapsed={elapsed:.1f}s")

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"Test Accuracy: {acc:.4f}  ({sum(a==b for a,b in zip(y_true,y_pred))}/{len(y_true)})")
    print(f"{'='*60}")
    print(f"\nClassification Report:\n{report}")
    return acc


def interactive_mode(classifier: IntentClassification):
    """Run an interactive loop for manual testing."""
    print("\n=== Interactive Intent Classification ===")
    print("Type a banking message and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in ("quit", "exit", "q"):
            break
        if not message:
            continue
        label = classifier(message)
        print(f"  -> Intent: {label}\n")
    print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Banking intent classification – inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="A single message to classify",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on the test set",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive classification mode",
    )
    args = parser.parse_args()

    classifier = IntentClassification(args.config)

    if args.message:
        label = classifier(args.message)
        print(f"\nMessage : {args.message}")
        print(f"Intent  : {label}")

    if args.evaluate:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        evaluate(classifier, cfg["data"]["test_file"])

    if args.interactive:
        interactive_mode(classifier)

    if not args.message and not args.evaluate and not args.interactive:
        print("\nNo action specified. Use --message, --evaluate, or --interactive.")
        print("Example:")
        print('  python scripts/inference.py --config configs/inference.yaml \\')
        print('      --message "I want to cancel my bank transfer"')


if __name__ == "__main__":
    main()
