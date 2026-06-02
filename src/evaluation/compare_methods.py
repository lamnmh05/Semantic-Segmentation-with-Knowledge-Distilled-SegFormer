"""
compare_methods.py – Visual comparison of multiple KD methods.

Produces a grid figure (N rows × 5 cols):
  (a) Image | (b) Ground Truth | (c) MLP | (d) BPKD | (e) Combine

Usage:
    python -m src.evaluation.compare_methods \
        --mlp_config     configs/eval_MLPFD.yml \
        --bpkd_config    configs/eval_BPKD.yml \
        --combine_config configs/eval_Combine.yml \
        --output_dir     eval_results/comparison \
        --num_images     10 \
        --random_subset
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.train import get_dataset, get_model
from src.evaluation.main_eval import load_eval_checkpoint


# ---------------------------------------------------------------------------
# Color palette (consistent across all methods)
# ---------------------------------------------------------------------------
def get_color_palette(num_classes=256):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[255] = [0, 0, 0]
    return colors


def decode_segmap(mask, colors, num_classes=150):
    """Convert label mask to RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        idx = mask == c
        if idx.any():
            rgb[idx] = colors[c]
    # ignore index
    rgb[mask == 255] = [0, 0, 0]
    return rgb


# ---------------------------------------------------------------------------
# Load config + model helper
# ---------------------------------------------------------------------------
def load_config_and_model(config_path, device, checkpoint_override=None):
    """Load yaml config, build student model, load checkpoint weights."""
    if not config_path.endswith(".yml") and not config_path.endswith(".yaml"):
        from transformers import SegformerForSemanticSegmentation
        print(f"Loading HF model directly: {config_path}")
        model = SegformerForSemanticSegmentation.from_pretrained(config_path, ignore_mismatched_sizes=True)
        model.to(device)
        model.eval()
        return model, None

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
# Inference on a single batch
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, images, target_size, device):
    """Run model inference and return prediction mask (numpy)."""
    images = images.to(device, non_blocking=True)
    logits = model(images)
    
    if hasattr(logits, 'logits'):
        logits = logits.logits
    elif isinstance(logits, tuple):
        logits = logits[0]
        
    logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    preds = torch.argmax(logits, dim=1)
    return preds[0].cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# De-normalize image tensor → numpy RGB
# ---------------------------------------------------------------------------
def tensor_to_image(image_tensor):
    """Convert normalised image tensor [1,3,H,W] → uint8 numpy [H,W,3]."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = image_tensor[0].cpu().numpy()
    img = (img * std + mean) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8).transpose(1, 2, 0)


# ---------------------------------------------------------------------------
# Draw the comparison grid
# ---------------------------------------------------------------------------
def draw_comparison_grid(rows, output_dir, filenames=None, dpi=200, no_grid=False):
    """
    rows: list of dicts with keys
          {'image', 'gt', 'mlp', 'bpkd', 'combine', 'teacher'}
          each value is an RGB numpy array [H,W,3].
    """
    num_rows = len(rows)
    col_labels = ["(a) Image", "(b) Ground Truth", "(c) MLP", "(d) BPKD", "(e) Combine", "(f) Teacher"]
    col_keys   = ["image",     "gt",               "mlp",     "bpkd",     "combine",     "teacher"]

    # Border colors for each column (R,G,B 0-1)
    border_colors = [
        (1.0, 0.0, 0.0),   # red – original image
        (0.0, 0.8, 0.0),   # green – GT
        (0.0, 0.6, 1.0),   # blue – MLP
        (1.0, 0.5, 0.0),   # orange – BPKD
        (0.8, 0.0, 0.8),   # magenta – Combine
        (0.5, 0.5, 0.5),   # gray – Teacher
    ]

    os.makedirs(output_dir, exist_ok=True)
    cell_h, cell_w = 2.2, 2.6
    
    if not no_grid:
        fig, axes = plt.subplots(
            num_rows, 6,
            figsize=(cell_w * 6, cell_h * num_rows + 0.6),
            gridspec_kw={"hspace": 0.08, "wspace": 0.04},
        )

        if num_rows == 1:
            axes = axes[np.newaxis, :]

        for r in range(num_rows):
            for c in range(6):
                ax = axes[r, c]
                img = rows[r][col_keys[c]]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])

                # Dashed coloured border (like the reference image)
                color = border_colors[c]
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.0)
                    spine.set_linestyle((0, (4, 3)))  # dashed

                # Column label on first row only
                if r == 0:
                    ax.set_title(col_labels[c], fontsize=10, fontweight="bold", pad=4)

        grid_path = os.path.join(output_dir, "comparison_grid.png")
        fig.savefig(grid_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"Saved comparison grid → {grid_path}")

    # Also save individual rows
    for r in range(num_rows):
        fig_row, axes_row = plt.subplots(1, 6, figsize=(cell_w * 6, cell_h))
        for c in range(6):
            ax = axes_row[c]
            ax.imshow(rows[r][col_keys[c]])
            ax.set_xticks([])
            ax.set_yticks([])
            color = border_colors[c]
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.0)
                spine.set_linestyle((0, (4, 3)))
            ax.set_title(col_labels[c], fontsize=9, fontweight="bold", pad=3)
            
        if filenames and r < len(filenames):
            base_name = os.path.splitext(filenames[r])[0]
            row_name = f"{base_name}_comparison.png"
        else:
            row_name = f"row_{r:04d}.png"
            
        row_path = os.path.join(output_dir, row_name)
        fig_row.savefig(row_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig_row)

    print(f"Saved {num_rows} individual row images → {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Visual comparison of KD methods")
    parser.add_argument("--mlp_config",     type=str, required=True, help="Config for MLP method")
    parser.add_argument("--bpkd_config",    type=str, required=True, help="Config for BPKD method")
    parser.add_argument("--combine_config", type=str, required=True, help="Config for Combine method")
    parser.add_argument("--teacher_name_or_path", type=str, default="nvidia/segformer-b4-finetuned-ade-512-512")
    parser.add_argument("--data_root",      type=str, default=None,  help="Override dataset path")
    parser.add_argument("--mlp_checkpoint",     type=str, default=None, help="Override MLP checkpoint path")
    parser.add_argument("--bpkd_checkpoint",    type=str, default=None, help="Override BPKD checkpoint path")
    parser.add_argument("--combine_checkpoint", type=str, default=None, help="Override Combine checkpoint path")
    parser.add_argument("--output_dir",     type=str, default="eval_results/comparison")
    parser.add_argument("--num_images",     type=int, default=10)
    parser.add_argument("--random_subset",  action="store_true",     help="Random sample instead of first N")
    parser.add_argument("--no_grid",        action="store_true",     help="Do not save the combined comparison grid image")
    parser.add_argument("--image_indices",  type=str, default=None,  help="Comma-separated indices, e.g. 0,5,12")
    parser.add_argument("--image_filenames",type=str, default=None,  help="Comma-separated filenames, e.g. ADE_val_00001234.jpg")
    parser.add_argument("--dpi",            type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----- Load 3 models -----
    print("\n=== Loading MLP model ===")
    model_mlp, cfg_mlp = load_config_and_model(args.mlp_config, device, args.mlp_checkpoint)

    print("\n=== Loading BPKD model ===")
    model_bpkd, cfg_bpkd = load_config_and_model(args.bpkd_config, device, args.bpkd_checkpoint)

    print("\n=== Loading Combine model ===")
    model_combine, cfg_combine = load_config_and_model(args.combine_config, device, args.combine_checkpoint)

    print("\n=== Loading Teacher model ===")
    model_teacher, _ = load_config_and_model(args.teacher_name_or_path, device, None)

    # ----- Build validation dataset (use config from MLP – same dataset) -----
    cfg = cfg_mlp
    if args.data_root:
        cfg["dataset"]["data_root"] = args.data_root

    val_split = "validation" if cfg["dataset"]["name"] == "ADE20K" else "val"
    val_dataset = get_dataset(cfg, split=val_split)

    # ----- Select indices -----
    all_indices = list(range(len(val_dataset)))

    if args.image_filenames:
        filenames = [x.strip() for x in args.image_filenames.split(",")]
        indices = []
        for fn in filenames:
            if hasattr(val_dataset, "images"):
                try:
                    idx = val_dataset.images.index(fn)
                    indices.append(idx)
                except ValueError:
                    print(f"Warning: Filename {fn} not found in validation set. Skipping.")
        print(f"Using matched indices from filenames: {indices}")
    elif args.image_indices:
        indices = [int(x.strip()) for x in args.image_indices.split(",")]
        print(f"Using manually specified indices: {indices}")
    elif args.random_subset:
        random.seed(42)
        indices = random.sample(all_indices, min(args.num_images, len(val_dataset)))
        print(f"Random subset (seed=42): {len(indices)} images")
    else:
        indices = all_indices[:args.num_images]
        print(f"First {len(indices)} images")

    subset = torch.utils.data.Subset(val_dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False,
        num_workers=cfg["dataset"].get("num_workers", 4),
        pin_memory=False,
    )

    # ----- Color palette -----
    num_classes = cfg["model"]["num_classes"]
    colors = get_color_palette()

    # ----- Inference -----
    print(f"\nRunning inference on {len(indices)} images …")
    rows = []

    for i, (images, labels) in enumerate(loader):
        target_size = labels.shape[-2:]

        img_rgb  = tensor_to_image(images)
        gt_np    = labels[0].cpu().numpy().astype(np.uint8)
        gt_rgb   = decode_segmap(gt_np, colors, num_classes)

        pred_mlp     = predict(model_mlp,     images, target_size, device)
        pred_bpkd    = predict(model_bpkd,    images, target_size, device)
        pred_combine = predict(model_combine, images, target_size, device)
        pred_teacher = predict(model_teacher, images, target_size, device)

        rows.append({
            "image":   img_rgb,
            "gt":      gt_rgb,
            "mlp":     decode_segmap(pred_mlp,     colors, num_classes),
            "bpkd":    decode_segmap(pred_bpkd,    colors, num_classes),
            "combine": decode_segmap(pred_combine, colors, num_classes),
            "teacher": decode_segmap(pred_teacher, colors, num_classes),
        })
        print(f"  [{i+1}/{len(indices)}] done (idx={indices[i]})")

    # ----- Extract filenames for saving -----
    filenames = []
    if hasattr(val_dataset, "images"):
        filenames = [val_dataset.images[idx] for idx in indices]

    # ----- Draw grid -----
    draw_comparison_grid(rows, args.output_dir, filenames=filenames, dpi=args.dpi, no_grid=args.no_grid)
    
    # ----- Save metadata -----
    import csv
    csv_path = os.path.join(args.output_dir, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Row_Image", "Dataset_Index", "Original_Filename"])
        for i, idx in enumerate(indices):
            filename = val_dataset.images[idx] if hasattr(val_dataset, "images") else "unknown"
            writer.writerow([f"row_{i:04d}.png", idx, filename])
    print(f"Saved metadata → {csv_path}")

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
