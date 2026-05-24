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

You can run the training script using the provided configs:

```bash
python main.py --config configs/AttnFD_ADE20k.yml
```

## Kaggle Execution
If you are running on Kaggle, you can clone the repository and run it directly within a Notebook cell:

```python
!git clone -b feat/source-feature https://github.com/lamnmh05/Semantic-Segmentation-with-Knowledge-Distilled-SegFormer.git
%cd Semantic-Segmentation-with-Knowledge-Distilled-SegFormer
!pip install -r requirements.txt

# Run the training script
!python main.py --config configs/AttnFD_ADE20k.yml
```

## Evaluation
Evaluation metrics (mIoU, FLOPs, Params, FPS) are automatically computed at the end of training or can be run separately using `src/eval.py`.
