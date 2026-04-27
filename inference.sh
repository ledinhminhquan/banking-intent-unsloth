#!/usr/bin/env bash
# ==============================================================================
# inference.sh – Run intent classification inference
# ==============================================================================
set -euo pipefail

CONFIG="configs/inference.yaml"

echo "=============================================="
echo " Banking77 Intent Classification – Inference"
echo "=============================================="

if [ $# -ge 1 ]; then
    MESSAGE="$*"
    echo ""
    echo "Classifying message: \"${MESSAGE}\""
    echo ""
    python scripts/inference.py --config "$CONFIG" --message "$MESSAGE"
else
    echo ""
    echo "Running demo predictions ..."
    echo ""
    python scripts/inference.py --config "$CONFIG" --message "I want to activate my new card"
    echo "---"
    python scripts/inference.py --config "$CONFIG" --message "Why was I charged an extra fee on my statement?"
    echo "---"
    python scripts/inference.py --config "$CONFIG" --message "My card payment is still pending after 3 days"
    echo "---"
    python scripts/inference.py --config "$CONFIG" --message "How do I change my PIN number?"
    echo "---"
    python scripts/inference.py --config "$CONFIG" --message "I lost my phone and need to secure my account"
    echo "---"
    python scripts/inference.py --config "$CONFIG" --message "What is the exchange rate for USD to EUR?"
    echo ""
    echo "=============================================="
    echo " Running evaluation on test set ..."
    echo "=============================================="
    python scripts/inference.py --config "$CONFIG" --evaluate
fi

echo ""
echo "=============================================="
echo " Inference complete!"
echo "=============================================="
