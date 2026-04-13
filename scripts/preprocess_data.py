"""
preprocess_data.py
------------------
Download the BANKING77 dataset from HuggingFace, sample a manageable subset,
perform basic text preprocessing, and save train/test splits plus a label map.
"""

import os
import json
import argparse
import re

import pandas as pd
from datasets import load_dataset


def clean_text(text: str) -> str:
    """Basic text normalization for banking queries."""
    text = text.strip()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def main(args):
    print("Loading BANKING77 dataset from HuggingFace …")
    dataset = load_dataset("PolyAI/banking77")

    label_names = dataset["train"].features["label"].names
    id2label = {str(i): name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    num_labels = len(label_names)

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    train_df["label_text"] = train_df["label"].map(lambda x: id2label[str(x)])
    test_df["label_text"] = test_df["label"].map(lambda x: id2label[str(x)])

    # ---- Sample a subset to fit available compute ----
    if args.train_samples_per_class > 0:
        train_df = (
            train_df.groupby("label", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=min(args.train_samples_per_class, len(g)),
                    random_state=args.seed,
                )
            )
            .reset_index(drop=True)
        )

    if args.test_samples_per_class > 0:
        test_df = (
            test_df.groupby("label", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=min(args.test_samples_per_class, len(g)),
                    random_state=args.seed,
                )
            )
            .reset_index(drop=True)
        )

    # ---- Clean text ----
    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    # ---- Shuffle ----
    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    label_map_path = os.path.join(args.output_dir, "label_map.json")

    train_df[["text", "label", "label_text"]].to_csv(train_path, index=False)
    test_df[["text", "label", "label_text"]].to_csv(test_path, index=False)

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {"id2label": id2label, "label2id": label2id, "num_labels": num_labels},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Number of intent labels : {num_labels}")
    print(f"Training samples        : {len(train_df)}")
    print(f"Test samples            : {len(test_df)}")
    print(f"Files saved to          : {args.output_dir}/")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {label_map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess BANKING77 dataset for intent classification"
    )
    parser.add_argument(
        "--train_samples_per_class",
        type=int,
        default=40,
        help="Max training samples per intent class (0 = use all)",
    )
    parser.add_argument(
        "--test_samples_per_class",
        type=int,
        default=10,
        help="Max test samples per intent class (0 = use all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sample_data",
        help="Directory to save processed CSV files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
