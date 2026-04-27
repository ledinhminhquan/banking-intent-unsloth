# Banking77 Intent Classification with Unsloth

Fine-tuning a large language model (**LLaMA 3.2 3B Instruct**) for **banking intent classification** on the [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) dataset using **Unsloth** (4-bit QLoRA).

> **Course**: CSC15012 – Applications of Natural Language Processing in Industry
> **Student**: Lê Đình Minh Quân – 23127460
> **University**: HCMUS – VNUHCM

---

## Project Structure

```text
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
└── README.md
```

---

## Quick Start

### 1. Environment Setup (Google Colab – recommended)

Open a **Colab notebook** with a **GPU runtime** (T4 / A100 / H100) and run:

```bash
# Install Unsloth
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

This downloads the BANKING77 dataset, samples a compute-friendly subset, applies text preprocessing, and saves CSV files under `sample_data/`.

**Current sampled split from this notebook run**
- Train: **3075**
- Test: **770**
- Number of intent labels: **77**

### 3. Training

```bash
python scripts/train.py --config configs/train.yaml
```

Or run the all-in-one script:

```bash
bash train.sh
```

**Key hyperparameters** (editable in `configs/train.yaml`):

| Parameter | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Learning rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Batch size | 8 x 4 (gradient accumulation) |
| Epochs | 3 |
| Max sequence length | 512 |
| Precision | bf16 (auto-detects fp16 fallback) |

The fine-tuned adapter is saved to `saved_model/`.

### 4. Inference

**Single message**
```bash
python scripts/inference.py \
    --config configs/inference.yaml \
    --message "I want to activate my new card"
```

**Evaluate the full test set**
```bash
python scripts/inference.py --config configs/inference.yaml --evaluate
```

**Interactive mode**
```bash
python scripts/inference.py --config configs/inference.yaml --interactive
```

**Or use the demo shell script**
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

| Metric | Value |
|---|---|
| Test Accuracy | _TBD_ |
| Macro Precision | _TBD_ |
| Macro Recall | _TBD_ |
| Macro F1 | _TBD_ |
| Training Loss | _TBD_ |
| Training Time | _TBD_ |

The full classification report is saved to `evaluation_results.json` after training.

---

## Links

📓 **Notebook link**: [Google Drive / Colab Notebook](https://drive.google.com/file/d/1dd4BxqryUvpkOvmDkm8-ZWor0LHpVSZu/view?usp=sharing)

📹 **Video link**: [Google Drive – Demo Video](https://drive.google.com/file/d/1D9hAiMirLMdjMijR-znzDAdYw2LGtdqP/view?usp=sharing)

The demo video should clearly show:
1. How the inference script is executed
2. At least one example input message
3. The predicted intent label
4. The final test accuracy

---

## References

- [BANKING77 Dataset](https://huggingface.co/datasets/PolyAI/banking77)
- [Unsloth](https://github.com/unslothai/unsloth)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

## License

This project is for educational purposes (CSC15012 – HCMUS).
