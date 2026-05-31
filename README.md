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
uv run main.py --config configs/combine.yml --data_path ADEChallengeData2016 --batch_size 32 --max_iters 40000 --lr 0.00006

uv run main.py --config configs/combine.yml --data_path ADEChallengeData2016 --batch_size 48 --max_iters 60000 --lr 0.00006
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
Evaluation metrics (mIoU, FLOPs, Params, FPS) are automatically computed at the end of training or can be run separately using `src/eval.py`.
