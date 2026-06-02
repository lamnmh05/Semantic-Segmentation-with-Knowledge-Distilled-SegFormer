"""
error_taxonomy.py – Phần E: Error Type Taxonomy.

Phân loại mỗi pixel lỗi (pred ≠ GT) thành 3 nhóm:
  1. Boundary Error: pixel nằm gần boundary (trong trimap)
  2. Misclassification Error: pixel xa boundary, nhầm class
  3. (bonus) tính tổng error pixel count

Tham khảo: "What's Outside the Intersection?" (WACV 2024)

Output:
  - Stacked bar chart: tỉ lệ % từng loại lỗi cho MLP / BPKD / Combine
  - CSV chi tiết
  - Console summary

Usage:
    python -m src.evaluation.error_taxonomy \
        --mlp_config     configs/eval_MLPFD.yml \
        --bpkd_config    configs/eval_BPKD.yml \
        --combine_config configs/eval_Combine.yml \
        --output_dir     eval_results/error_taxonomy \
        --num_images     500 \
        --random_subset
"""

import os
import sys
import argparse
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.train import get_dataset, get_model
from src.evaluation.main_eval import load_eval_checkpoint


# ---------------------------------------------------------------------------
# Create boundary mask from GT
# ---------------------------------------------------------------------------
def compute_boundary_mask(gt_np, num_classes, boundary_width=5):
    """
    Compute binary boundary mask from GT label map.

    Uses morphological dilation - erosion to find boundary pixels.
    Returns binary mask: 1 = boundary pixel, 0 = body pixel.
    """
    import cv2

    h, w = gt_np.shape
    boundary = np.zeros((h, w), dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (boundary_width, boundary_width))

    # Process each class
    for c in range(num_classes):
        class_mask = (gt_np == c).astype(np.uint8)
        if class_mask.sum() == 0:
            continue

        dilated = cv2.dilate(class_mask, kernel, iterations=1)
        eroded = cv2.erode(class_mask, kernel, iterations=1)

        class_boundary = dilated - eroded
        boundary = np.maximum(boundary, class_boundary)

    return boundary


# ---------------------------------------------------------------------------
# Classify error pixels
# ---------------------------------------------------------------------------
def classify_errors(pred_np, gt_np, boundary_mask, ignore_index=255):
    """
    Classify each error pixel into:
      - boundary_error: pixel is wrong AND near a boundary
      - misclass_error: pixel is wrong AND NOT near a boundary (body region)

    Returns dict with counts.
    """
    valid = gt_np != ignore_index
    error = (pred_np != gt_np) & valid
    correct = (pred_np == gt_np) & valid

    error_at_boundary = error & (boundary_mask == 1)
    error_at_body = error & (boundary_mask == 0)

    total_valid = valid.sum()
    total_error = error.sum()
    total_correct = correct.sum()
    n_boundary_error = error_at_boundary.sum()
    n_body_error = error_at_body.sum()

    return {
        "total_valid": int(total_valid),
        "total_correct": int(total_correct),
        "total_error": int(total_error),
        "boundary_error": int(n_boundary_error),
        "body_error": int(n_body_error),
    }


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(config_path, device):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    num_classes = cfg["model"]["num_classes"]
    model = get_model(cfg["model"]["student"], num_classes)
    model.to(device)
    checkpoint_path = cfg["model"].get("checkpoint")
    load_eval_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Run error taxonomy for one model on entire loader
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_error_taxonomy(model, loader, device, num_classes, boundary_width=5):
    """Run inference and accumulate error type counts."""
    model.eval()
    totals = {
        "total_valid": 0,
        "total_correct": 0,
        "total_error": 0,
        "boundary_error": 0,
        "body_error": 0,
    }

    for images, labels in tqdm(loader, desc="Error taxonomy"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1)

        pred_np = preds[0].cpu().numpy().astype(np.int32)
        gt_np = labels[0].cpu().numpy().astype(np.int32)

        boundary_mask = compute_boundary_mask(gt_np, num_classes, boundary_width)
        counts = classify_errors(pred_np, gt_np, boundary_mask)

        for k in totals:
            totals[k] += counts[k]

    return totals


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_error_taxonomy(results, output_dir):
    """
    Stacked bar chart: boundary error % vs body error % for each method.
    Also includes correct pixels %.
    """
    methods = list(results.keys())
    n = len(methods)

    boundary_pct = []
    body_pct = []
    correct_pct = []

    for m in methods:
        r = results[m]
        total = r["total_valid"]
        boundary_pct.append(r["boundary_error"] / total * 100)
        body_pct.append(r["body_error"] / total * 100)
        correct_pct.append(r["total_correct"] / total * 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ----- Plot 1: Error breakdown (only error pixels) -----
    ax1 = axes[0]
    boundary_err_pct_of_err = []
    body_err_pct_of_err = []
    for m in methods:
        r = results[m]
        total_err = r["total_error"]
        boundary_err_pct_of_err.append(r["boundary_error"] / total_err * 100 if total_err > 0 else 0)
        body_err_pct_of_err.append(r["body_error"] / total_err * 100 if total_err > 0 else 0)

    x = np.arange(n)
    w = 0.5

    bars1 = ax1.bar(x, boundary_err_pct_of_err, w, label="Boundary Error", color="#E74C3C", alpha=0.85)
    bars2 = ax1.bar(x, body_err_pct_of_err, w, bottom=boundary_err_pct_of_err,
                    label="Misclassification Error (Body)", color="#3498DB", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.set_ylabel("% of Total Error Pixels", fontsize=11)
    ax1.set_title("Error Type Breakdown\n(% among error pixels)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (b, m_val) in enumerate(zip(boundary_err_pct_of_err, body_err_pct_of_err)):
        ax1.text(i, b / 2, f"{b:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        ax1.text(i, b + m_val / 2, f"{m_val:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    # ----- Plot 2: Overall pixel distribution -----
    ax2 = axes[1]
    bars_c = ax2.bar(x, correct_pct, w, label="Correct", color="#2ECC71", alpha=0.85)
    bars_be = ax2.bar(x, boundary_pct, w, bottom=correct_pct,
                      label="Boundary Error", color="#E74C3C", alpha=0.85)
    bottom2 = [c + b for c, b in zip(correct_pct, boundary_pct)]
    bars_me = ax2.bar(x, body_pct, w, bottom=bottom2,
                      label="Misclass Error", color="#3498DB", alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=11)
    ax2.set_ylabel("% of Total Valid Pixels", fontsize=11)
    ax2.set_title("Overall Pixel Distribution\n(correct vs error types)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3)

    # Error rate labels
    for i in range(n):
        err_total = boundary_pct[i] + body_pct[i]
        ax2.text(i, 50, f"Err: {err_total:.1f}%", ha="center", fontsize=9,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    plt.tight_layout()
    path = os.path.join(output_dir, "error_taxonomy.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved error taxonomy plot → {path}")


def plot_error_reduction(results, output_dir):
    """
    Grouped bar: total error count / boundary error / body error for each method.
    Shows absolute reduction.
    """
    methods = list(results.keys())
    n = len(methods)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n)
    w = 0.25

    boundary = [results[m]["boundary_error"] / 1e6 for m in methods]
    body = [results[m]["body_error"] / 1e6 for m in methods]
    total_err = [results[m]["total_error"] / 1e6 for m in methods]

    ax.bar(x - w, total_err, w, label="Total Error", color="#95A5A6", alpha=0.85)
    ax.bar(x, boundary, w, label="Boundary Error", color="#E74C3C", alpha=0.85)
    ax.bar(x + w, body, w, label="Body (Misclass) Error", color="#3498DB", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Error Pixels (millions)", fontsize=11)
    ax.set_title("Absolute Error Pixel Count", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for i in range(n):
        ax.text(i - w, total_err[i] + 0.1, f"{total_err[i]:.2f}M", ha="center", fontsize=8)
        ax.text(i, boundary[i] + 0.1, f"{boundary[i]:.2f}M", ha="center", fontsize=8)
        ax.text(i + w, body[i] + 0.1, f"{body[i]:.2f}M", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "error_absolute_count.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved error count plot → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Error Type Taxonomy analysis")
    parser.add_argument("--mlp_config",     type=str, required=True)
    parser.add_argument("--bpkd_config",    type=str, required=True)
    parser.add_argument("--combine_config", type=str, required=True)
    parser.add_argument("--data_root",      type=str, default=None)
    parser.add_argument("--output_dir",     type=str, default="eval_results/error_taxonomy")
    parser.add_argument("--num_images",     type=int, default=500)
    parser.add_argument("--random_subset",  action="store_true")
    parser.add_argument("--boundary_width", type=int, default=5, help="Trimap boundary width in pixels")
    parser.add_argument("--num_workers",    type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    with open(args.mlp_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.data_root:
        cfg["dataset"]["data_root"] = args.data_root

    val_split = "validation" if cfg["dataset"]["name"] == "ADE20K" else "val"
    val_dataset = get_dataset(cfg, split=val_split)
    num_classes = cfg["model"]["num_classes"]

    # Select subset
    indices = list(range(len(val_dataset)))
    if args.num_images < len(val_dataset):
        if args.random_subset:
            random.seed(42)
            indices = random.sample(indices, args.num_images)
        else:
            indices = indices[:args.num_images]
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    print(f"Dataset: {len(val_dataset)} images, {num_classes} classes")
    print(f"Boundary width: {args.boundary_width}px")

    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    # Evaluate each method
    configs = {
        "MLP-FD":  args.mlp_config,
        "BPKD":    args.bpkd_config,
        "Combine": args.combine_config,
    }

    results = {}
    for name, config_path in configs.items():
        print(f"\n{'='*50}")
        print(f"  Error Taxonomy: {name}")
        print(f"{'='*50}")
        model, _ = load_model(config_path, device)
        totals = run_error_taxonomy(model, loader, device, num_classes, args.boundary_width)
        results[name] = totals

        err_rate = totals["total_error"] / totals["total_valid"] * 100
        b_rate = totals["boundary_error"] / totals["total_error"] * 100 if totals["total_error"] > 0 else 0
        m_rate = totals["body_error"] / totals["total_error"] * 100 if totals["total_error"] > 0 else 0

        print(f"  Total pixels: {totals['total_valid']:,}")
        print(f"  Correct: {totals['total_correct']:,} ({100 - err_rate:.2f}%)")
        print(f"  Error: {totals['total_error']:,} ({err_rate:.2f}%)")
        print(f"    ├─ Boundary: {totals['boundary_error']:,} ({b_rate:.1f}% of errors)")
        print(f"    └─ Misclass: {totals['body_error']:,} ({m_rate:.1f}% of errors)")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*80}")
    print(f"  ERROR TAXONOMY SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Method':<10} | {'Error Rate':>10} | {'Boundary Err':>12} | {'Misclass Err':>12} | {'Boundary %':>10}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for name, r in results.items():
        err_rate = r["total_error"] / r["total_valid"] * 100
        b_rate = r["boundary_error"] / r["total_error"] * 100 if r["total_error"] > 0 else 0
        print(f"  {name:<10} | {err_rate:>9.2f}% | {r['boundary_error']:>12,} | {r['body_error']:>12,} | {b_rate:>9.1f}%")
    print(f"{'='*80}\n")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(args.output_dir, "error_taxonomy_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Total Valid", "Total Correct", "Total Error",
                        "Boundary Error", "Body (Misclass) Error",
                        "Error Rate (%)", "Boundary % of Errors", "Body % of Errors"])
        for name, r in results.items():
            err_rate = r["total_error"] / r["total_valid"] * 100
            b_pct = r["boundary_error"] / r["total_error"] * 100 if r["total_error"] > 0 else 0
            m_pct = r["body_error"] / r["total_error"] * 100 if r["total_error"] > 0 else 0
            writer.writerow([name, r["total_valid"], r["total_correct"], r["total_error"],
                            r["boundary_error"], r["body_error"],
                            f"{err_rate:.2f}", f"{b_pct:.1f}", f"{m_pct:.1f}"])
    print(f"Saved CSV → {csv_path}")

    # Plots
    plot_error_taxonomy(results, args.output_dir)
    plot_error_reduction(results, args.output_dir)

    print("\n✅ Error taxonomy analysis done!")


if __name__ == "__main__":
    main()
