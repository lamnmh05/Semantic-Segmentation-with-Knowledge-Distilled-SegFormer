# Semantic Segmentation with Knowledge Distilled SegFormer

This repository contains the implementation for Semantic Segmentation using Knowledge Distillation with SegFormer. 
It supports multiple distillation methods (FitNet, AttnFD, MLPFD, BPKD, Combine) and datasets (ADE20K, COCOStuff).

## Features
- **Models**: SegFormer (Teacher/Student)
- **Datasets**: ADE20K, COCOStuff
- **Distillation Methods**: 
  - FitNet (Feature-based Distillation)
  - AttnFD (Attention-based Feature Distillation)
  - MLPFD (MLP-based Feature Distillation)
  - BPKD (Boundary Privileged Knowledge Distillation)
  - Combine (MLPFD + BPKD)

## Vast.ai Setup

If you are running on Vast.ai, use the following commands:
```bash
git clone --branch feat/source-combine https://github.com/lamnmh05/Semantic-Segmentation-with-Knowledge-Distilled-SegFormer.git

cd Semantic-Segmentation-with-Knowledge-Distilled-SegFormer

wget -O ade20k-dataset.zip https://www.kaggle.com/api/v1/datasets/download/awsaf49/ade20k-dataset

unzip ade20k-dataset.zip

pip install uv

uv sync
```

Run training with the Combine config:
```bash
uv run main.py --config configs/combine_weight.yml --data_path ADEChallengeData2016 --batch_size 16

uv run main.py --config configs/BPKD_ADE20k.yml --data_path ADEChallengeData2016 --batch_size 16

uv run main.py --config configs/MLP_ADE20k.yml --data_path ADEChallengeData2016 --batch_size 16

uv run main.py --config configs/Segformer_ADE20k.yml --data_path ADEChallengeData2016 --batch_size 16

uv run main.py --config configs/Segformer_B4_ADE20k.yml --data_path ADEChallengeData2016 --batch_size 16
```

If the extracted dataset folder is not in the repository root, update `--data_path` to the actual `ADEChallengeData2016` path.

## Kaggle Execution
If you are running on Kaggle, you can clone the repository and run it directly within a Notebook cell:

```python
!git clone --branch feat/source-combine https://github.com/lamnmh05/Semantic-Segmentation-with-Knowledge-Distilled-SegFormer.git
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

