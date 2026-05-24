import argparse

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from src.dataloaders.ade20k import ADE20KDataset
from src.dataloaders.coco_stuff import CocoStuff
from src.distillers.attn_fd import AttnFD
from src.distillers.fit_net import FitNet
from src.engine.trainer import Trainer


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataset(cfg, split):
    dataset_name = cfg["dataset"]["name"]
    data_root = cfg["dataset"]["data_root"]
    img_size = cfg["dataset"]["img_size"]
    augment = cfg["dataset"].get("augment", False)
    scale_range = tuple(cfg["dataset"].get("scale_range", [0.5, 2.0]))
    flip_prob = cfg["dataset"].get("flip_prob", 0.5)

    common = dict(
        data_root=data_root,
        img_size=img_size,
        augment=augment,
        scale_range=scale_range,
        flip_prob=flip_prob,
    )
    if dataset_name == "ADE20K":
        return ADE20KDataset(split=split, **common)
    if dataset_name == "COCOStuff":
        return CocoStuff(split=split, **common)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_model(model_name, num_classes):
    print(f"Instantiating model {model_name} with {num_classes} classes...")

    class DummyModel(torch.nn.Module):
        def __init__(self, name, num_classes):
            super().__init__()
            self.name = name
            self.num_classes = num_classes
            self.conv = torch.nn.Conv2d(3, num_classes, 1)

        def extract_feature(self, x):
            h8, w8 = x.size(2) // 8, x.size(3) // 8
            ch = 256 if "b5" in self.name else 64
            feats = [
                torch.randn(x.size(0), ch, h8, w8, device=x.device) for _ in range(4)
            ]
            attens = [
                torch.randn(x.size(0), ch, h8, w8, device=x.device) for _ in range(4)
            ]
            return feats, attens, self.conv(x)

        def forward(self, x):
            return self.conv(x)

    return DummyModel(model_name, num_classes)


def build_optimizer(distiller, cfg):
    train_cfg = cfg["train"]
    opt_type = train_cfg.get("optimizer", "AdamW")
    lr = train_cfg["lr"]
    weight_decay = train_cfg.get("weight_decay", 0.01)
    head_lr_mult = train_cfg.get("head_lr_mult", 10.0)
    connector_lr_mult = train_cfg.get("connector_lr_mult", 10.0)

    backbone_params = []
    head_params = []
    extra_params = []

    for name, param in distiller.student.named_parameters():
        if not param.requires_grad:
            continue
        if "decode" in name or "head" in name or "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    if hasattr(distiller, "get_connector_parameters"):
        extra_params = distiller.get_connector_parameters()
    elif hasattr(distiller, "get_regressor_parameters"):
        extra_params = distiller.get_regressor_parameters()

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr_mult": 1.0})
    if head_params:
        param_groups.append({"params": head_params, "lr_mult": head_lr_mult})
    if extra_params:
        param_groups.append({"params": extra_params, "lr_mult": connector_lr_mult})

    if not param_groups:
        param_groups = [{"params": distiller.get_learnable_parameters(), "lr_mult": 1.0}]

    for group in param_groups:
        group["lr"] = lr * group.get("lr_mult", 1.0)

    betas = tuple(train_cfg.get("betas", [0.9, 0.999]))
    if opt_type.lower() == "sgd":
        return optim.SGD(
            param_groups,
            lr=lr,
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=train_cfg.get("nesterov", False),
        )
    return optim.AdamW(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for SegFormer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--data_path", type=str, default=None, help="Override path to dataset root")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
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

    teacher_ckpt = cfg["model"].get("teacher_checkpoint")
    if teacher_ckpt:
        state = torch.load(teacher_ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        teacher.load_state_dict(state, strict=False)
        print(f"Loaded teacher weights from {teacher_ckpt}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    distill_method = cfg["distill"]["method"]
    if distill_method == "FitNet":
        distiller = FitNet(student, teacher, cfg)
    elif distill_method == "AttnFD":
        distiller = AttnFD(student, teacher, cfg)
    else:
        raise ValueError(f"Unknown distillation method: {distill_method}")

    distiller = distiller.to(device)
    optimizer = build_optimizer(distiller, cfg)

    Trainer(distiller, train_loader, val_loader, optimizer, device, cfg).train()


if __name__ == "__main__":
    main()
