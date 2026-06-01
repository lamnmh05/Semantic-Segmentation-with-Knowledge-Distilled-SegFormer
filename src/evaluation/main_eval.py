import os
import sys
import re
import argparse
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.train import get_dataset, get_model
from src.eval import Evaluator
from src.evaluation.error_analysis import ErrorAnalyzer


def remap_state_dict(state_dict, target_keys):
    target_keys_set = set(target_keys)
    new_state = {}

    for k, v in state_dict.items():
        if k in target_keys_set:
            new_state[k] = v
            continue

        new_k = k

        if "encoder.patch_embeddings" in new_k:
            new_k = re.sub(r"encoder\.patch_embeddings\.(\d+)\.", r"stages.\1.patch_embeddings.", new_k)
        if "encoder.block" in new_k:
            new_k = re.sub(r"encoder\.block\.(\d+)\.(\d+)\.", r"stages.\1.blocks.\2.", new_k)
        if "encoder.layer_norm" in new_k:
            new_k = re.sub(r"encoder\.layer_norm\.(\d+)\.", r"stages.\1.layer_norm.", new_k)
        if "encoder.norm" in new_k:
            new_k = re.sub(r"encoder\.norm\.(\d+)\.", r"stages.\1.layer_norm.", new_k)

        new_k = new_k.replace("attention.self.query", "attention.q_proj")
        new_k = new_k.replace("attention.self.key", "attention.k_proj")
        new_k = new_k.replace("attention.self.value", "attention.v_proj")
        new_k = new_k.replace("attention.self.sr", "attention.sequence_reduction.sequence_reduction")
        new_k = new_k.replace("attention.self.layer_norm", "attention.sequence_reduction.layer_norm")
        new_k = new_k.replace("attention.output.dense", "attention.o_proj")

        new_k = new_k.replace("mlp.dense1", "mlp.fc1")
        new_k = new_k.replace("mlp.dense2", "mlp.fc2")
        new_k = new_k.replace("layer_norm_1", "layernorm_before")
        new_k = new_k.replace("layer_norm_2", "layernorm_after")

        if "decode_head.linear_c" in new_k:
            new_k = re.sub(r"decode_head\.linear_c\.(\d+)\.", r"decode_head.linear_projections.\1.", new_k)

        if new_k in target_keys_set:
            new_state[new_k] = v
        else:
            new_state[k] = v

    return new_state


def load_eval_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    target_keys = list(model.state_dict().keys())
    missing_before = [k for k in target_keys if k not in state]

    if len(missing_before) > len(target_keys) * 0.5:
        print(f"Detected old checkpoint format ({len(missing_before)}/{len(target_keys)} keys mismatched). Remapping...")
        state = remap_state_dict(state, target_keys)

    info = model.load_state_dict(state, strict=False)
    missing_after = info.missing_keys
    unexpected_after = info.unexpected_keys

    if missing_after:
        print(f"Missing keys ({len(missing_after)}): {missing_after[:5]}")
    if unexpected_after:
        print(f"Unexpected keys ({len(unexpected_after)}): {unexpected_after[:5]}")

    loaded = len(target_keys) - len(missing_after)
    print(f"Loaded weights: {checkpoint_path} ({loaded}/{len(target_keys)} keys)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--eval_mode", type=str, choices=["full", "subset"], default="full")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--random_subset", action="store_true")
    parser.add_argument("--save_vis", action="store_true")
    return parser.parse_args()


def save_pred_image(pred_np, colors, save_path):
    from PIL import Image
    h, w = pred_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in np.unique(pred_np):
        if c == 255:
            continue
        mask = pred_np == c
        rgb[mask] = colors[c]
    Image.fromarray(rgb).save(save_path)


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.data_root:
        cfg["dataset"]["data_root"] = args.data_root

    val_split = "validation" if cfg["dataset"]["name"] == "ADE20K" else "val"
    val_dataset = get_dataset(cfg, split=val_split)

    indices = list(range(len(val_dataset)))
    if args.eval_mode == "subset":
        if args.random_subset:
            random.seed(42)
            indices = random.sample(indices, min(args.num_images, len(val_dataset)))
        else:
            indices = indices[:args.num_images]
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
        print(f"Subset: {len(val_dataset)} images")
    else:
        print(f"Full: {len(val_dataset)} images")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["dataset"].get("num_workers", 4),
        pin_memory=False
    )

    num_classes = cfg["model"]["num_classes"]
    model = get_model(cfg["model"]["student"], num_classes)
    model.to(device)

    checkpoint_path = args.checkpoint or cfg["model"].get("checkpoint")
    load_eval_checkpoint(model, checkpoint_path, device)

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    label_names = None
    if hasattr(model, 'model') and hasattr(model.model.config, 'id2label'):
        label_names = [model.model.config.id2label[i] for i in range(num_classes)]

    analyzer = ErrorAnalyzer(num_classes, args.output_dir, label_names)
    evaluator = Evaluator(num_classes)

    pred_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

    print("Starting evaluation...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(val_loader)):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)

            evaluator.update(preds, labels)

            pred_np = preds[0].cpu().numpy().astype(np.uint8)
            save_pred_image(pred_np, colors, os.path.join(pred_dir, f"pred_{i:04d}.png"))

            if args.save_vis:
                analyzer.save_visualizations(images, labels, preds, f"img_{i:04d}")

    miou, ious = evaluator.compute_miou()
    print(f"\nResult: mIoU = {miou:.2f}%")

    analyzer.save_per_class_metrics(ious, miou)
    analyzer.save_confusion_matrix(evaluator.hist)
    print(f"Done! Results at: {args.output_dir}")
    print(f"Prediction images at: {pred_dir}")


if __name__ == "__main__":
    main()
