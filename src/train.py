import torch
import torch.optim as optim
import yaml

from src.dataloaders.ade20k import ADE20KDataset
from src.dataloaders.coco_stuff import CocoStuff
from src.models.segformer import build_segformer


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    print(f"Loaded weights: {checkpoint_path}")


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


def _parse_model_entry(model_cfg, role):
    entry = model_cfg.get(role)
    if isinstance(entry, dict):
        name = entry.get("name") or entry.get("model") or entry.get("id")
        pretrained = entry.get("pretrained")
        pretrained_encoder = entry.get("pretrained_encoder") or entry.get("encoder_pretrained")
        return name, pretrained, pretrained_encoder

    name = entry
    pretrained = model_cfg.get(f"{role}_pretrained")
    pretrained_encoder = model_cfg.get(f"{role}_pretrained_encoder") or model_cfg.get(
        f"{role}_encoder_pretrained"
    )
    return name, pretrained, pretrained_encoder


def get_model(model_or_cfg, num_classes=None, role=None):
    if role is None:
        if num_classes is None:
            raise ValueError("num_classes is required when role is not provided")
        return build_segformer(model_or_cfg, num_classes)

    model_cfg = model_or_cfg
    name, pretrained, pretrained_encoder = _parse_model_entry(model_cfg, role)
    if not name:
        raise ValueError(f"Missing model name for role='{role}'")

    ignore_index = int(model_cfg.get("ignore_index", 255))
    return build_segformer(
        name,
        model_cfg["num_classes"],
        pretrained=pretrained,
        pretrained_encoder=pretrained_encoder,
        ignore_index=ignore_index,
    )


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
