"""
quantitative_eval.py – Quantitative Overview cho 3 KD methods.

Chạy eval trên TOÀN BỘ validation set, tính:
  - mIoU (%)
  - Pixel Accuracy (%)
  - Mean Accuracy (mAcc) (%)

Xuất bảng so sánh dạng CSV + console table.

Usage:
    python -m src.evaluation.quantitative_eval \
        --mlp_config     configs/eval_MLPFD.yml \
        --bpkd_config    configs/eval_BPKD.yml \
        --combine_config configs/eval_Combine.yml \
        --output_dir     eval_results/quantitative
"""

import os
import sys
import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.train import get_dataset, get_model
from src.evaluation.main_eval import load_eval_checkpoint
from src.eval import fast_hist, per_class_iou


# ---------------------------------------------------------------------------
# Extended Evaluator with Pixel Acc + mAcc
# ---------------------------------------------------------------------------
class ExtendedEvaluator:
    """Evaluator that computes mIoU, Pixel Accuracy, and mean class Accuracy."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, gt):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        mask = gt != 255
        self.hist += fast_hist(gt[mask].flatten(), pred[mask].flatten(), self.num_classes)

    def compute_all(self):
        """Return dict with mIoU, pixel_acc, mAcc, per_class_iou, per_class_acc."""
        # Per-class IoU
        ious = per_class_iou(self.hist)
        miou = float(np.nanmean(ious) * 100)

        # Pixel Accuracy = sum(diag) / sum(all)
        total_pixels = self.hist.sum()
        correct_pixels = np.diag(self.hist).sum()
        pixel_acc = float(correct_pixels / (total_pixels + 1e-6) * 100)

        # Mean class Accuracy = mean of (TP_i / (TP_i + FN_i))
        class_correct = np.diag(self.hist)
        class_total = self.hist.sum(axis=1)
        class_acc = class_correct / (class_total + 1e-6)
        macc = float(np.nanmean(class_acc) * 100)

        return {
            "mIoU": miou,
            "pixel_acc": pixel_acc,
            "mAcc": macc,
            "per_class_iou": ious * 100,
            "per_class_acc": class_acc * 100,
            "hist": self.hist,
        }


# ---------------------------------------------------------------------------
# Run eval for a single model
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model, loader, device, num_classes):
    model.eval()
    evaluator = ExtendedEvaluator(num_classes)

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1)
        evaluator.update(preds, labels)

    return evaluator.compute_all()


# ---------------------------------------------------------------------------
# Load config + model
# ---------------------------------------------------------------------------
def load_model(config_path, device, checkpoint_override=None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    num_classes = cfg["model"]["num_classes"]
    model = get_model(cfg["model"]["student"], num_classes)
    model.to(device)
    checkpoint_path = checkpoint_override or cfg["model"].get("checkpoint")
    load_eval_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Quantitative evaluation of KD methods")
    parser.add_argument("--mlp_config",     type=str, required=True)
    parser.add_argument("--bpkd_config",    type=str, required=True)
    parser.add_argument("--combine_config", type=str, required=True)
    parser.add_argument("--data_root",      type=str, default=None,  help="Override dataset root path")
    parser.add_argument("--mlp_checkpoint",     type=str, default=None, help="Override MLP checkpoint path")
    parser.add_argument("--bpkd_checkpoint",    type=str, default=None, help="Override BPKD checkpoint path")
    parser.add_argument("--combine_checkpoint", type=str, default=None, help="Override Combine checkpoint path")
    parser.add_argument("--output_dir",     type=str, default="eval_results/quantitative")
    parser.add_argument("--num_workers",    type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    methods = {
        "MLP-FD":  args.mlp_config,
        "BPKD":    args.bpkd_config,
        "Combine": args.combine_config,
    }
    checkpoint_overrides = {
        "MLP-FD":  args.mlp_checkpoint,
        "BPKD":    args.bpkd_checkpoint,
        "Combine": args.combine_checkpoint,
    }

    # Load first config for dataset
    with open(args.mlp_config, "r", encoding="utf-8") as f:
        cfg_base = yaml.safe_load(f)
    if args.data_root:
        cfg_base["dataset"]["data_root"] = args.data_root

    val_split = "validation" if cfg_base["dataset"]["name"] == "ADE20K" else "val"
    val_dataset = get_dataset(cfg_base, split=val_split)
    num_classes = cfg_base["model"]["num_classes"]

    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )
    print(f"Validation set: {len(val_dataset)} images, {num_classes} classes\n")

    # Evaluate each method
    results = {}
    all_per_class = {}

    for name, config_path in methods.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"  Config: {config_path}")
        print(f"{'='*60}")

        model, _ = load_model(config_path, device, checkpoint_overrides.get(name))
        res = evaluate_model(model, loader, device, num_classes)
        results[name] = res
        all_per_class[name] = res["per_class_iou"]

        print(f"  mIoU: {res['mIoU']:.2f}%  |  Pixel Acc: {res['pixel_acc']:.2f}%  |  mAcc: {res['mAcc']:.2f}%")

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  QUANTITATIVE OVERVIEW — ADE20K Validation")
    print(f"{'='*70}")
    print(f"  {'Method':<12} | {'mIoU (%)':<10} | {'Pixel Acc (%)':<14} | {'mAcc (%)':<10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*14}-+-{'-'*10}")
    for name, res in results.items():
        print(f"  {name:<12} | {res['mIoU']:>8.2f}  | {res['pixel_acc']:>12.2f}  | {res['mAcc']:>8.2f}")
    print(f"{'='*70}\n")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # CSV summary
    csv_path = os.path.join(args.output_dir, "quantitative_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "mIoU (%)", "Pixel Acc (%)", "mAcc (%)"])
        for name, res in results.items():
            writer.writerow([name, f"{res['mIoU']:.2f}", f"{res['pixel_acc']:.2f}", f"{res['mAcc']:.2f}"])
    print(f"Saved summary → {csv_path}")

    # Per-class IoU CSV (for Part B)
    per_class_csv = os.path.join(args.output_dir, "per_class_iou.csv")
    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Class_ID"] + list(methods.keys())
        writer.writerow(header)
        for c in range(num_classes):
            row = [c] + [f"{all_per_class[name][c]:.2f}" for name in methods]
            writer.writerow(row)
    print(f"Saved per-class IoU → {per_class_csv}")

    # Save confusion matrices
    for name, res in results.items():
        np.save(os.path.join(args.output_dir, f"confusion_matrix_{name}.npy"), res["hist"])

    print("\nQuantitative evaluation done!")


if __name__ == "__main__":
    main()
