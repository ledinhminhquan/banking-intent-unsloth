#!/usr/bin/env bash
# ==============================================================================
# inference.sh – Run intent classification inference
# ==============================================================================
set -euo pipefail

CONFIG="configs/inference.yaml"

echo "=============================================="
echo " Banking77 Intent Classification – Inference"
echo "=============================================="

# --- Option 1: Single message (default demo) ---
if [ $# -ge 1 ]; then
    MESSAGE="$*"
    echo ""
    echo "Classifying message: \"${MESSAGE}\""
    echo ""
    python scripts/inference.py --config "$CONFIG" --message "$MESSAGE"
else
    # Demo with several example messages
    echo ""
    echo "Running demo predictions …"
    echo ""

    python scripts/inference.py --config "$CONFIG" \
        --message "I want to activate my new card"

    echo "---"
    python scripts/inference.py --config "$CONFIG" \
        --message "Why was I charged an extra fee on my statement?"

    echo "---"
    python scripts/inference.py --config "$CONFIG" \
        --message "My card payment is still pending after 3 days"

    echo "---"
    python scripts/inference.py --config "$CONFIG" \
        --message "How do I change my PIN number?"

    echo "---"
    python scripts/inference.py --config "$CONFIG" \
        --message "I lost my phone and need to secure my account"

    # Also run full test-set evaluation
    echo ""
    echo "=============================================="
    echo " Running evaluation on test set …"
    echo "=============================================="
    python scripts/inference.py --config "$CONFIG" --evaluate
fi

echo ""
echo "=============================================="
echo " Inference complete!"
echo "=============================================="
