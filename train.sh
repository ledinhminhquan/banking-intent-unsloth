#!/usr/bin/env bash
# ==============================================================================
# train.sh – End-to-end pipeline: preprocess data -> fine-tune model
# ==============================================================================
set -euo pipefail

echo "=============================================="
echo " Banking77 Intent Classification – Training"
echo "=============================================="

echo ""
echo "[Step 1/2] Preprocessing BANKING77 dataset ..."
python scripts/preprocess_data.py \
    --train_samples_per_class 40 \
    --test_samples_per_class 10 \
    --output_dir sample_data \
    --seed 42

echo ""
echo "[Step 2/2] Fine-tuning model with Unsloth ..."
python scripts/train.py --config configs/train.yaml

echo ""
echo "=============================================="
echo " Training pipeline complete!"
echo "=============================================="
