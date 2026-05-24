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

```bash
pip install -r requirements.txt
```

## Training

You can run the training script using the provided configs:

```bash
python main.py --config configs/AttnFD_ADE20k.yml
```

## Kaggle Execution
If you are running on Kaggle, simply upload this repository and run the `main.py` script. The script is configured to parse the config arguments correctly.

```python
!python main.py --config configs/AttnFD_ADE20k.yml
```

## Evaluation
Evaluation metrics (mIoU, FLOPs, Params, FPS) are automatically computed at the end of training or can be run separately using `src/eval.py`.
