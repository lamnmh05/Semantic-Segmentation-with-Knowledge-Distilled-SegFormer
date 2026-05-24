import torch
import torch.optim as optim
import yaml

from src.dataloaders.ade20k import ADE20KDataset
from src.dataloaders.coco_stuff import CocoStuff

import json
from huggingface_hub import hf_hub_download, login
from transformers import SegformerConfig, SegformerForSemanticSegmentation


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


def get_model(model_name, num_classes):
    name_map = {
        "segformer_b0": "nvidia/mit-b0",
        "segformer_b1": "nvidia/mit-b1",
        "segformer_b2": "nvidia/mit-b2",
        "segformer_b3": "nvidia/mit-b3",
        "segformer_b4": "nvidia/mit-b4",
        "segformer_b5": "nvidia/mit-b5",
    }
    hf_model_name = name_map.get(model_name.lower(), model_name)
    print(f"Instantiating model {model_name} (HF repo: {hf_model_name}) with {num_classes} classes...")

    # Login to Hugging Face
    try:
        import os
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        else:
            print("No HF_TOKEN found in environment. Please login manually if required.")
    except Exception as e:
        print(f"HuggingFace login failed: {e}")

    class SegformerWrapper(torch.nn.Module):
        def __init__(self, name, num_classes):
            super().__init__()
            self.name = name
            self.num_classes = num_classes

            try:
                id2label_path = hf_hub_download(repo_id="huggingface/label-files", filename="ade20k-id2label.json", repo_type="dataset")
                with open(id2label_path, "r") as f:
                    id2label = json.load(f)
                id2label = {int(k): v for k, v in id2label.items()}
                label2id = {v: k for k, v in id2label.items()}
            except Exception as e:
                print(f"Failed to load label mapping: {e}")
                id2label = {i: str(i) for i in range(num_classes)}
                label2id = {str(i): i for i in range(num_classes)}

            config = SegformerConfig.from_pretrained(hf_model_name)
            config.num_labels = num_classes
            config.ignore_index = 255
            config.id2label = id2label
            config.label2id = label2id
            config.output_hidden_states = True

            self.model = SegformerForSemanticSegmentation.from_pretrained(hf_model_name, config=config, ignore_mismatched_sizes=True)

        def extract_feature(self, x):
            outputs = self.model(x, output_hidden_states=True, return_dict=True)
            feats = list(outputs.hidden_states)
            attens = list(outputs.hidden_states)
            logits = outputs.logits
            return feats, attens, logits

        def forward(self, x):
            return self.model(x).logits

    return SegformerWrapper(model_name, num_classes)


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
