"""
per_class_analysis.py – Phần B: Per-Class IoU Analysis.

Đọc file per_class_iou.csv từ Part A (quantitative_eval.py) và tạo:
  1. Bar chart per-class IoU (top-30 classes)
  2. Bảng top-10 classes Combine cải thiện nhiều nhất
  3. Bảng top-10 classes Combine giảm nhiều nhất
  4. Histogram phân bố IoU improvement

Usage (đọc CSV có sẵn):
    python -m src.evaluation.per_class_analysis \
        --per_class_csv eval_results/quantitative/per_class_iou.csv \
        --output_dir    eval_results/per_class

Usage (chạy eval trực tiếp):
    python -m src.evaluation.per_class_analysis \
        --mlp_config     configs/eval_MLPFD.yml \
        --bpkd_config    configs/eval_BPKD.yml \
        --combine_config configs/eval_Combine.yml \
        --output_dir     eval_results/per_class
"""

import os
import sys
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 9

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# ---------------------------------------------------------------------------
# Load per-class IoU from CSV
# ---------------------------------------------------------------------------
def load_per_class_csv(csv_path):
    """Return dict: {method_name: np.array of IoU per class}."""
    methods = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        method_names = header[1:]
        for name in method_names:
            methods[name] = []
        for row in reader:
            for i, name in enumerate(method_names):
                methods[name].append(float(row[i + 1]))
    for name in methods:
        methods[name] = np.array(methods[name])
    return methods, method_names


# ---------------------------------------------------------------------------
# Load ADE20K class names
# ---------------------------------------------------------------------------
def get_ade20k_class_names():
    """Return list of 150 ADE20K class names (simplified)."""
    # Top ADE20K class names — official list
    names = [
        "wall", "building", "sky", "floor", "tree",
        "ceiling", "road", "bed", "windowpane", "grass",
        "cabinet", "sidewalk", "person", "earth", "door",
        "table", "mountain", "plant", "curtain", "chair",
        "car", "water", "painting", "sofa", "shelf",
        "house", "sea", "mirror", "rug", "field",
        "armchair", "seat", "fence", "desk", "rock",
        "wardrobe", "lamp", "bathtub", "railing", "cushion",
        "base", "box", "column", "signboard", "chest of drawers",
        "counter", "sand", "sink", "skyscraper", "fireplace",
        "refrigerator", "grandstand", "path", "stairs", "runway",
        "case", "pool table", "pillow", "screen door", "stairway",
        "river", "bridge", "bookcase", "blind", "coffee table",
        "toilet", "flower", "book", "hill", "bench",
        "countertop", "stove", "palm", "kitchen island", "computer",
        "swivel chair", "boat", "bar", "arcade machine", "hovel",
        "bus", "towel", "light", "truck", "tower",
        "chandelier", "awning", "streetlight", "booth", "television",
        "airplane", "dirt track", "apparel", "pole", "land",
        "bannister", "escalator", "ottoman", "bottle", "buffet",
        "poster", "stage", "van", "ship", "fountain",
        "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
        "stool", "barrel", "basket", "waterfall", "tent",
        "bag", "minibike", "cradle", "oven", "ball",
        "food", "step", "tank", "trade name", "microwave",
        "pot", "animal", "bicycle", "lake", "dishwasher",
        "screen", "blanket", "sculpture", "hood", "sconce",
        "vase", "traffic light", "tray", "ashcan", "fan",
        "pier", "crt screen", "plate", "monitor", "bulletin board",
        "shower", "radiator", "glass", "clock", "flag",
    ]
    # Pad if needed
    while len(names) < 150:
        names.append(f"class_{len(names)}")
    return names[:150]


# ---------------------------------------------------------------------------
# Plot 1: Bar chart per-class IoU (top-N classes by Combine IoU)
# ---------------------------------------------------------------------------
def plot_per_class_bars(methods, method_names, class_names, output_dir, top_n=30):
    """Bar chart showing per-class IoU for top-N classes (sorted by Combine IoU desc)."""
    combine_key = method_names[-1]  # Assume Combine is last
    combine_iou = methods[combine_key]

    # Sort by Combine IoU descending, take top N
    sorted_idx = np.argsort(combine_iou)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(20, 7))
    x = np.arange(top_n)
    width = 0.25
    colors_bar = ["#4A90D9", "#E8753A", "#2ECC71"]

    for i, name in enumerate(method_names):
        vals = methods[name][sorted_idx]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors_bar[i], alpha=0.85, edgecolor='black', linewidth=0.5)
        # Add values on top of bars
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8, rotation=90)

    ax.set_xlabel("Class", fontsize=12, fontweight='bold')
    ax.set_ylabel("IoU (%)", fontsize=12, fontweight='bold')
    ax.set_title(f"Per-Class IoU — Top {top_n} Classes (sorted by {combine_key})", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels([class_names[i].upper() for i in sorted_idx], rotation=45, ha="right", fontsize=9)
    
    # Place legend outside or upper right
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis="y", linestyle='--', alpha=0.6)
    
    # Auto adjust y-limit so text doesn't get cut
    max_val = max([max(methods[n][sorted_idx]) for n in method_names])
    ax.set_ylim(0, min(100, max_val + 15))

    plt.tight_layout()
    path = os.path.join(output_dir, "per_class_iou_top30.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart → {path}")


# ---------------------------------------------------------------------------
# Plot 2: Full 150 classes heatmap-style
# ---------------------------------------------------------------------------
def plot_full_heatmap(methods, method_names, class_names, output_dir):
    """Horizontal grouped bar for ALL 150 classes."""
    combine_key = method_names[-1]
    sorted_idx = np.argsort(methods[combine_key])[::-1]

    fig, ax = plt.subplots(figsize=(12, 35))
    y = np.arange(150)
    height = 0.28
    colors_bar = ["#4A90D9", "#E8753A", "#2ECC71"]

    for i, name in enumerate(method_names):
        vals = methods[name][sorted_idx]
        ax.barh(y + i * height, vals, height, label=name, color=colors_bar[i], alpha=0.85, edgecolor='none')

    ax.set_ylabel("Class", fontsize=12, fontweight='bold')
    ax.set_xlabel("IoU (%)", fontsize=12, fontweight='bold')
    ax.set_title("Per-Class IoU — All 150 Classes", fontsize=16, fontweight="bold", pad=20)
    ax.set_yticks(y + height)
    ax.set_yticklabels([class_names[i].upper() for i in sorted_idx], fontsize=7)
    ax.legend(loc="lower right", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle='--', alpha=0.6)

    plt.tight_layout()
    path = os.path.join(output_dir, "per_class_iou_all150.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved full heatmap → {path}")


# ---------------------------------------------------------------------------
# Plot 3: IoU improvement histogram (Combine vs best of others)
# ---------------------------------------------------------------------------
def plot_improvement_histogram(methods, method_names, output_dir):
    """Histogram of IoU difference: Combine - max(MLP, BPKD)."""
    combine_key = method_names[-1]
    other_keys = method_names[:-1]

    best_other = np.maximum(*[methods[k] for k in other_keys])
    delta = methods[combine_key] - best_other

    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ['#E74C3C' if d < 0 else '#2ECC71' for d in delta]

    sorted_idx = np.argsort(delta)
    bars = ax.bar(range(len(delta)), delta[sorted_idx], color=[colors[i] for i in sorted_idx], alpha=0.85, edgecolor='black', linewidth=0.2)

    ax.axhline(y=0, color='black', linewidth=1.2)
    ax.set_xlabel("Classes (sorted by improvement)", fontsize=12, fontweight='bold')
    ax.set_ylabel("IoU Δ (Combine − best of MLP/BPKD)", fontsize=12, fontweight='bold')
    ax.set_title("Per-Class IoU Improvement by Combine", fontsize=16, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle='--', alpha=0.6)

    improved = (delta > 0).sum()
    degraded = (delta < 0).sum()
    unchanged = (delta == 0).sum()
    
    # Highlight text box with statistics
    stats_text = f"Improved: {improved} classes\nDegraded: {degraded} classes\nUnchanged: {unchanged} classes"
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9F9", edgecolor="#BDC3C7", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(output_dir, "iou_improvement_histogram.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved improvement histogram → {path}")


# ---------------------------------------------------------------------------
# Table: Top-10 improved & degraded classes
# ---------------------------------------------------------------------------
def save_top_classes_table(methods, method_names, class_names, output_dir, top_n=10):
    """Save CSV: top-N improved and top-N degraded classes."""
    combine_key = method_names[-1]
    other_keys = method_names[:-1]

    best_other = np.maximum(*[methods[k] for k in other_keys])
    delta = methods[combine_key] - best_other

    # Top improved
    improved_idx = np.argsort(delta)[::-1][:top_n]
    # Top degraded
    degraded_idx = np.argsort(delta)[:top_n]

    csv_path = os.path.join(output_dir, "top_classes_analysis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["=== TOP-10 IMPROVED BY COMBINE ==="])
        writer.writerow(["Rank", "Class_ID", "Class_Name"] + [f"{n} IoU" for n in method_names] + ["Delta"])
        for rank, idx in enumerate(improved_idx, 1):
            row = [rank, idx, class_names[idx]]
            row += [f"{methods[n][idx]:.2f}" for n in method_names]
            row += [f"{delta[idx]:+.2f}"]
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(["=== TOP-10 DEGRADED BY COMBINE ==="])
        writer.writerow(["Rank", "Class_ID", "Class_Name"] + [f"{n} IoU" for n in method_names] + ["Delta"])
        for rank, idx in enumerate(degraded_idx, 1):
            row = [rank, idx, class_names[idx]]
            row += [f"{methods[n][idx]:.2f}" for n in method_names]
            row += [f"{delta[idx]:+.2f}"]
            writer.writerow(row)

    print(f"Saved top classes analysis → {csv_path}")

    # Print to console
    print(f"\n{'='*70}")
    print(f"  TOP-10 CLASSES IMPROVED BY COMBINE")
    print(f"{'='*70}")
    print(f"  {'Class':<20} | {'MLP-FD':>8} | {'BPKD':>8} | {'Combine':>8} | {'Δ':>6}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    for idx in improved_idx:
        print(f"  {class_names[idx]:<20} | {methods[method_names[0]][idx]:>7.2f}% | {methods[method_names[1]][idx]:>7.2f}% | {methods[method_names[2]][idx]:>7.2f}% | {delta[idx]:>+5.2f}")

    print(f"\n  TOP-10 CLASSES DEGRADED BY COMBINE")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    for idx in degraded_idx:
        print(f"  {class_names[idx]:<20} | {methods[method_names[0]][idx]:>7.2f}% | {methods[method_names[1]][idx]:>7.2f}% | {methods[method_names[2]][idx]:>7.2f}% | {delta[idx]:>+5.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Per-class IoU analysis")
    parser.add_argument("--per_class_csv", type=str, default=None,
                        help="Path to per_class_iou.csv from quantitative_eval.py")
    parser.add_argument("--mlp_config",     type=str, default=None)
    parser.add_argument("--bpkd_config",    type=str, default=None)
    parser.add_argument("--combine_config", type=str, default=None)
    parser.add_argument("--data_root",      type=str, default=None)
    parser.add_argument("--output_dir",     type=str, default="eval_results/per_class")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    class_names = get_ade20k_class_names()

    if args.per_class_csv:
        # Mode 1: Read from CSV (from quantitative_eval.py output)
        print(f"Loading per-class IoU from: {args.per_class_csv}")
        methods, method_names = load_per_class_csv(args.per_class_csv)
    elif args.mlp_config and args.bpkd_config and args.combine_config:
        # Mode 2: Run eval directly
        import torch
        import torch.nn.functional as F
        import yaml
        from tqdm import tqdm
        from src.evaluation.quantitative_eval import load_model, ExtendedEvaluator

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(args.mlp_config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if args.data_root:
            cfg["dataset"]["data_root"] = args.data_root
        val_split = "validation" if cfg["dataset"]["name"] == "ADE20K" else "val"
        val_dataset = get_dataset(cfg, split=val_split)
        num_classes = cfg["model"]["num_classes"]

        loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=cfg["dataset"].get("num_workers", 4), pin_memory=False,
        )

        configs = {"MLP-FD": args.mlp_config, "BPKD": args.bpkd_config, "Combine": args.combine_config}
        methods = {}
        method_names = list(configs.keys())

        for name, config_path in configs.items():
            print(f"\nEvaluating {name}...")
            model, _ = load_model(config_path, device)
            evaluator = ExtendedEvaluator(num_classes)
            with torch.no_grad():
                for images, labels in tqdm(loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    preds = torch.argmax(logits, dim=1)
                    evaluator.update(preds, labels)
            res = evaluator.compute_all()
            methods[name] = res["per_class_iou"]
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print("Error: Provide --per_class_csv OR all three --*_config args.")
        sys.exit(1)

    # Generate all outputs
    print("\nGenerating plots...")
    plot_per_class_bars(methods, method_names, class_names, args.output_dir, top_n=30)
    plot_full_heatmap(methods, method_names, class_names, args.output_dir)
    plot_improvement_histogram(methods, method_names, args.output_dir)
    save_top_classes_table(methods, method_names, class_names, args.output_dir)

    print("\nPer-class analysis done!")


if __name__ == "__main__":
    main()
