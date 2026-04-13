# Banking77 Intent Classification with Unsloth

Fine-tuning a large language model (LLaMA 3.2 3B) for **banking intent classification** on the [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) dataset using **Unsloth** (4-bit QLoRA).

> **Course**: CSC15012 – Applications of Natural Language Processing in Industry  
> **Student**: Lê Đình Minh Quân – 23127460  
> **University**: HCMUS – VNUHCM

---

## Project Structure

```
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py    # Download & preprocess BANKING77
│   ├── train.py              # Fine-tune with Unsloth + LoRA
│   └── inference.py          # Standalone inference class
├── configs/
│   ├── train.yaml            # Training hyperparameters
│   └── inference.yaml        # Inference configuration
├── sample_data/
│   ├── train.csv             # Sampled training set
│   ├── test.csv              # Sampled test set
│   └── label_map.json        # Intent label mapping
├── train.sh                  # One-click training script
├── inference.sh              # One-click inference / demo script
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md                 # This file
```

---

## Quick Start

### 1. Environment Setup (Google Colab – recommended)

Open a **Colab notebook** with a **GPU runtime** (T4 / A100 / H100) and run:

```bash
# Install Unsloth (optimized for Colab)
pip install unsloth

# Install remaining dependencies
pip install datasets pandas scikit-learn pyyaml

# Clone this repository
git clone https://github.com/ledinhminhquan/banking-intent-unsloth.git
cd banking-intent-unsloth
```

### 2. Data Preparation

```bash
python scripts/preprocess_data.py \
    --train_samples_per_class 40 \
    --test_samples_per_class 10 \
    --output_dir sample_data \
    --seed 42
```

This downloads the BANKING77 dataset, samples a subset (≈ 3,080 train / 770 test), applies text preprocessing, and saves CSV files under `sample_data/`.

### 3. Training

```bash
python scripts/train.py --config configs/train.yaml
```

Or use the all-in-one script:

```bash
bash train.sh
```

**Key hyperparameters** (editable in `configs/train.yaml`):

| Parameter | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Learning rate | 2 × 10⁻⁴ |
| Optimizer | AdamW 8-bit |
| Batch size | 8 × 4 (gradient accumulation) |
| Epochs | 3 |
| Max sequence length | 512 |
| Precision | bf16 (auto-detects fp16 fallback) |

The fine-tuned LoRA adapter is saved to `saved_model/`.

### 4. Inference

**Single message:**
```bash
python scripts/inference.py \
    --config configs/inference.yaml \
    --message "I want to activate my new card"
```

**Test-set evaluation:**
```bash
python scripts/inference.py --config configs/inference.yaml --evaluate
```

**Interactive mode:**
```bash
python scripts/inference.py --config configs/inference.yaml --interactive
```

**Or use the demo script:**
```bash
bash inference.sh
```

### 5. Inference Class API

```python
from scripts.inference import IntentClassification

classifier = IntentClassification("configs/inference.yaml")

label = classifier("I need to change my PIN")
print(label)  # e.g. "change_pin"
```

---

## Results

<!-- Update these numbers after training -->
| Metric | Value |
|---|---|
| Test Accuracy | _TBD_ |
| Training Loss | _TBD_ |
| Training Time | _TBD_ |

Full classification report is saved to `evaluation_results.json` after training.

---

## Video Demonstration

<!-- Replace with your actual Google Drive link -->
📹 **Video link**: [Google Drive – Demo Video](https://drive.google.com/YOUR_VIDEO_LINK_HERE)

The video shows:
1. Running the inference script
2. Example input messages and predicted intents
3. Test-set accuracy

---

## References

- [BANKING77 Dataset](https://huggingface.co/datasets/PolyAI/banking77) – Casanueva et al. (2020)
- [Unsloth](https://github.com/unslothai/unsloth) – Efficient LLM fine-tuning
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) – Hu et al. (2021)

---

## License

This project is for educational purposes (CSC15012 – HCMUS).
