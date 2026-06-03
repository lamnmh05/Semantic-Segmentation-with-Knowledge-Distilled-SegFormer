# Semantic Segmentation with Knowledge Distilled SegFormer

This repository contains the implementation for Semantic Segmentation using Knowledge Distillation with SegFormer. 
It supports multiple distillation methods and datasets (ADE20K, COCOStuff).

## Project Structure
```text
.
├── configs/                  # Configuration files for different methods/datasets
├── docs/                     # Documentation files (Error Analysis, etc.)
├── eval_results/             # Generated evaluation outputs and visualizations
├── src/                      # Source code directory
│   ├── dataloaders/          # Data loading scripts (ADE20K, COCOStuff, Augmentations)
│   ├── distillers/           # Knowledge Distillation methods (MLP, BPKD, Combine, etc.)
│   ├── engine/               # Training engine and LR scheduling
│   ├── evaluation/           # Evaluation and Error Analysis scripts
│   ├── models/               # SegFormer architectures
│   ├── eval.py               # Evaluation entry point for Trainer
│   └── train.py              # Training entry point
├── main.py                   # Main execution script for training/evaluating
├── uv.lock                   # Dependencies lockfile (uv)
└── requirements.txt          # Python dependencies
```

## Preparation & Setup

### 1. Environment Setup
We recommend using [uv](https://github.com/astral-sh/uv) or `pip` to install dependencies.
```bash
# Clone the repository
git clone https://github.com/lamnmh05/Semantic-Segmentation-with-Knowledge-Distilled-SegFormer.git
cd Semantic-Segmentation-with-Knowledge-Distilled-SegFormer

# Install dependencies
pip install -r requirements.txt
# OR using uv
uv sync
```

### 2. Dataset Preparation (ADE20K)
Download the ADE20K dataset and extract it into the project directory (or anywhere on your machine).
```bash
wget -O ade20k-dataset.zip https://www.kaggle.com/api/v1/datasets/download/awsaf49/ade20k-dataset
unzip ade20k-dataset.zip
```
Ensure the folder `ADEChallengeData2016` exists. You will pass its path using the `--data_path` argument.

---

## Training

To train the student model using a specific distillation method, use `main.py` with the corresponding config file.

**Train with Combine (MLPFD + BPKD):**
```bash
python main.py --config configs/combine_weight.yml --data_path path/to/ADEChallengeData2016 --batch_size 16
```

**Train with other methods:**
```bash
python main.py --config configs/BPKD_ADE20k.yml --data_path path/to/ADEChallengeData2016
python main.py --config configs/MLP_ADE20k.yml --data_path path/to/ADEChallengeData2016
python main.py --config configs/Segformer_ADE20k.yml --data_path path/to/ADEChallengeData2016
```

*Note: You can override hyperparameters directly from the CLI (e.g., `--epochs 50`, `--lr 0.0001`, `--batch_size 8`).*

---

## Evaluation & Error Analysis

### 1. Inline Evaluation
During training, the model is automatically evaluated on the validation set at regular intervals (defined by `eval_interval` in config).

To run evaluation only on a specific checkpoint:
```bash
python main.py --config configs/BPKD_ADE20k.yml --eval_only --student_ckpt path/to/checkpoint.pth
```

### 2. Error Analysis Suite
We provide a comprehensive suite for Quantitative, Qualitative, and Per-Class Error Analysis.

**Run full error analysis on a trained model:**
```bash
python -m src.evaluation.main_eval \
  --config configs/eval_BPKD.yml \
  --eval_mode full \
  --output_dir eval_results/BPKD \
  --save_vis
```

**Arguments:**
| Argument | Description | Default |
|---|---|---|
| `--config` | Path to evaluation config file | *required* |
| `--checkpoint` | Override checkpoint path | from config |
| `--data_root` | Override dataset root path | from config |
| `--eval_mode` | `full` (all images) or `subset` (n images) | `full` |
| `--save_vis` | Save side-by-side visualization overlays | `false` |

**Outputs generated in `eval_results/`:**
- `predictions/` — Raw segmentation prediction images.
- `visualizations/` — Side-by-side comparison (Original | GT | Prediction | Overlay).
- `per_class_metrics.csv` — Detailed Per-class IoU scores.
- `confusion_matrix.png` — Normalized confusion matrix chart.
- `error_taxonomy.png` — Error taxonomy distribution.

---

## Cloud / Kaggle Execution
If running on Kaggle or Colab, clone the repo and execute directly from a notebook cell:

```python
!git clone https://github.com/lamnmh05/Semantic-Segmentation-with-Knowledge-Distilled-SegFormer.git
%cd Semantic-Segmentation-with-Knowledge-Distilled-SegFormer
!pip install -r requirements.txt

# Run training
!python main.py --config configs/combine_weight.yml --data_path /kaggle/input/datasets/awsaf49/ade20k-dataset/ADEChallengeData2016 --epochs 50
```