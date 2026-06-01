# Semantic Segmentation with Knowledge Distilled SegFormer

This repository contains the implementation for Semantic Segmentation using Knowledge Distillation with SegFormer. 
It supports multiple distillation methods (FitNet, AttnFD) and datasets (ADE20K, COCOStuff).

## Features
- **Models**: SegFormer (Teacher/Student)
- **Datasets**: ADE20K, COCOStuff
- **Distillation Methods**: 
  - FitNet (Feature-based Distillation)
  - AttnFD (Attention-based Feature Distillation)

## Installation

First, clone the repository (specifically the `feat/source-feature` branch) and navigate into the project directory:
```bash
git clone -b feat/source-feature https://github.com/lamnmh05/Semantic-Segmentation-with-Knowledge-Distilled-SegFormer.git
cd Semantic-Segmentation-with-Knowledge-Distilled-SegFormer
```

Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training

You can run the training script using the provided configs and optionally override parameters like dataset path, epochs, and batch size:

```bash
python main.py --config configs/AttnFD_ADE20k.yml --data_path ./datasets/ADE20K --epochs 50 --batch_size 8
```

## Kaggle Execution
If you are running on Kaggle, you can clone the repository and run it directly within a Notebook cell:

```python
!git clone -b feat/source-feature https://github.com/lamnmh05/Semantic-Segmentation-with-Knowledge-Distilled-SegFormer.git
%cd Semantic-Segmentation-with-Knowledge-Distilled-SegFormer
!pip install -r requirements.txt

# Run the training script with dataset path and epoch overrides
!python main.py --config configs/AttnFD_ADE20k.yml --data_path /kaggle/input/datasets/awsaf49/ade20k-dataset/ADEChallengeData2016 --epochs 50 --batch_size 8
```

## Evaluation

### Error Analysis Evaluation

Use `src/evaluation/main_eval.py` to evaluate trained student models on the validation set with detailed error analysis.

**Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--config` | Path to evaluation config file | *required* |
| `--checkpoint` | Override checkpoint path | from config |
| `--data_root` | Override dataset root path | from config |
| `--output_dir` | Directory to save results | `eval_results` |
| `--eval_mode` | `full` (all images) or `subset` (n images) | `full` |
| `--num_images` | Number of images for subset mode | `100` |
| `--random_subset` | Randomly select subset images | `false` |
| `--save_vis` | Save visualization overlays | `false` |

**Evaluate MLPFD:**
```bash
python -m src.evaluation.main_eval \
  --config configs/eval_MLPFD.yml \
  --eval_mode full \
  --output_dir eval_results/MLPFD \
  --save_vis
```

**Evaluate BPKD:**
```bash
python -m src.evaluation.main_eval \
  --config configs/eval_BPKD.yml \
  --eval_mode full \
  --output_dir eval_results/BPKD \
  --save_vis
```

**Subset evaluation (quick test):**
```bash
python -m src.evaluation.main_eval \
  --config configs/eval_BPKD.yml \
  --eval_mode subset \
  --num_images 50 \
  --random_subset \
  --output_dir eval_results/BPKD_subset \
  --save_vis
```

**Override paths (Kaggle/Colab):**
```bash
python -m src.evaluation.main_eval \
  --config configs/eval_BPKD.yml \
  --data_root /kaggle/input/ade20k-dataset/ADEChallengeData2016 \
  --checkpoint /kaggle/working/BPKD_best.pth \
  --eval_mode full \
  --output_dir eval_results/BPKD
```

**Output:**
- `predictions/` — Segmentation prediction images (color-mapped)
- `visualizations/` — Side-by-side comparison (Original | GT | Prediction | Overlay), only with `--save_vis`
- `per_class_metrics.csv` — Per-class IoU scores
- `confusion_matrix.png` — Normalized confusion matrix
- `confusion_matrix.npy` — Raw confusion matrix data

### Inline Evaluation (during training)

Evaluation metrics (mIoU, FLOPs, Params, FPS) are automatically computed during training at every `eval_interval` iterations, or can be triggered with `--eval_only`:

```bash
python main.py --config configs/BPKD_ADE20k.yml --eval_only --student_ckpt path/to/checkpoint.pth
```

