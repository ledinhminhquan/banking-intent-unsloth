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

BANKING77_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay",
    "atm_support", "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire",
    "card_acceptance", "card_arrival", "card_delivery_estimate",
    "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
    "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits",
    "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer",
    "fiat_currency_support", "freeze_card",
    "get_disposable_virtual_card", "get_physical_card",
    "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
    "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money",
    "Refund_not_showing_up", "request_refund", "reverted_card_payment?",
    "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up",
    "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


def _load_banking77():
    """Load BANKING77 with multiple fallbacks for different datasets versions."""
    # Strategy 1: regular dataset load with trust_remote_code
    try:
        ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
        label_names = ds["train"].features["label"].names
        print("  Loaded via default method (trust_remote_code=True).")
        return ds, label_names
    except Exception as e:
        print(f"  Default load failed: {e}")

    # Strategy 2: auto-converted parquet branch
    try:
        ds = load_dataset(
            "PolyAI/banking77",
            revision="refs/convert/parquet",
            trust_remote_code=True,
        )
        try:
            label_names = ds["train"].features["label"].names
        except (AttributeError, KeyError):
            label_names = BANKING77_LABELS
        print("  Loaded via parquet revision.")
        return ds, label_names
    except Exception as e:
        print(f"  Parquet revision failed: {e}")

    # Strategy 3: direct parquet files from the Hub
    try:
        base = "hf://datasets/PolyAI/banking77@refs%2Fconvert%2Fparquet"
        ds = load_dataset(
            "parquet",
            data_files={
                "train": f"{base}/default/train/0000.parquet",
                "test": f"{base}/default/test/0000.parquet",
            },
        )
        print("  Loaded via direct parquet URLs.")
        return ds, BANKING77_LABELS
    except Exception as e:
        print(f"  Direct parquet failed: {e}")

    raise RuntimeError(
        "Could not load BANKING77. Try upgrading datasets or re-running with internet access."
    )


def clean_text(text: str) -> str:
    """Basic text normalization for banking queries."""
    text = str(text).strip()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _sample_per_class(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    if max_per_class <= 0:
        return df.copy()

    sampled = (
        df.groupby("label", group_keys=False)
        .apply(
            lambda g: g.sample(
                n=min(max_per_class, len(g)),
                random_state=seed,
            )
        )
        .reset_index(drop=True)
    )
    return sampled


def main(args):
    print("Loading BANKING77 dataset from HuggingFace ...")
    dataset, label_names = _load_banking77()

    id2label = {str(i): name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    num_labels = len(label_names)

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    train_df["label_text"] = train_df["label"].map(lambda x: id2label[str(x)])
    test_df["label_text"] = test_df["label"].map(lambda x: id2label[str(x)])

    original_train_counts = train_df["label"].value_counts().sort_index()
    original_test_counts = test_df["label"].value_counts().sort_index()

    # ---- Sample a subset to fit available compute ----
    train_df = _sample_per_class(train_df, args.train_samples_per_class, args.seed)
    test_df = _sample_per_class(test_df, args.test_samples_per_class, args.seed)

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

    requested_train = args.train_samples_per_class
    requested_test = args.test_samples_per_class

    under_train = int((original_train_counts < requested_train).sum()) if requested_train > 0 else 0
    under_test = int((original_test_counts < requested_test).sum()) if requested_test > 0 else 0

    print(f"Number of intent labels : {num_labels}")
    print(f"Requested train/class   : {requested_train}")
    print(f"Requested test/class    : {requested_test}")
    print(f"Training samples        : {len(train_df)}")
    print(f"Test samples            : {len(test_df)}")
    if requested_train > 0:
        print(f"Classes with < {requested_train} train samples in original split : {under_train}")
    if requested_test > 0:
        print(f"Classes with < {requested_test} test samples in original split  : {under_test}")
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
