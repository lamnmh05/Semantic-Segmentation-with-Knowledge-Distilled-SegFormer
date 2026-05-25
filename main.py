import sys
import argparse
import torch
from torch.utils.data import DataLoader

from src.train import load_config, get_dataset, get_model, build_optimizer, load_checkpoint
from src.distillers.attn_fd import AttnFD
from src.distillers.fit_net import FitNet
from src.engine.trainer import Trainer
from src.eval import run_student_evaluation

def main():
    if len(sys.argv) == 1:
        print("No arguments provided. Please run with --config path/to/config.yml")
        print("Example: python main.py --config configs/AttnFD_ADE20k.yml --data_path /kaggle/input/dataset --epochs 50")

    parser = argparse.ArgumentParser(description="Knowledge Distillation for SegFormer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--data_path", type=str, default=None, help="Override path to dataset root")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate student on val set")
    parser.add_argument("--student_ckpt", type=str, default=None, help="Student checkpoint for eval_only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # Overrides
    if args.data_path is not None:
        cfg["dataset"]["data_root"] = args.data_path
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
        if "max_iters" in cfg["train"]:
            del cfg["train"]["max_iters"]
    if args.batch_size is not None:
        cfg["dataset"]["batch_size"] = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_split = "training" if cfg["dataset"]["name"] == "ADE20K" else "train"
    val_split = "validation" if cfg["dataset"]["name"] == "ADE20K" else "val"

    train_dataset = get_dataset(cfg, split=train_split)
    val_dataset = get_dataset(cfg, split=val_split)

    loader_kwargs = dict(
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=cfg["dataset"].get("pin_memory", True),
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    num_classes = cfg["model"]["num_classes"]
    teacher = get_model(cfg["model"]["teacher"], num_classes).to(device)
    student = get_model(cfg["model"]["student"], num_classes).to(device)

    load_checkpoint(teacher, cfg["model"].get("teacher_checkpoint"), device)
    student_ckpt = args.student_ckpt or cfg["model"].get("student_checkpoint")
    load_checkpoint(student, student_ckpt, device)

    if args.eval_only:
        img_size = cfg["dataset"]["img_size"]
        h, w = (img_size[0], img_size[1]) if isinstance(img_size, (list, tuple)) else (img_size, img_size)
        run_student_evaluation(
            student,
            val_loader,
            device,
            num_classes,
            student_name=cfg["model"].get("student", "student"),
            input_size=(1, 3, h, w),
        )
        return

    distill_method = cfg["distill"]["method"]
    if distill_method == "FitNet":
        distiller = FitNet(student, teacher, cfg)
    elif distill_method == "AttnFD":
        distiller = AttnFD(student, teacher, cfg)
    else:
        raise ValueError(f"Unknown distillation method: {distill_method}")

    distiller = distiller.to(device)
    optimizer = build_optimizer(distiller, cfg)
    if torch.cuda.device_count() > 1:
        distiller = torch.nn.DataParallel(distiller)

    Trainer(distiller, train_loader, val_loader, optimizer, device, cfg).train()

if __name__ == "__main__":
    main()
