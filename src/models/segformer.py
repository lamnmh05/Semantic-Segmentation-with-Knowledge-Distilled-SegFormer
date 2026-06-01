import json
import os
from typing import Optional, Tuple

import torch
from huggingface_hub import hf_hub_download, login
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel

SEGFORMER_NAME_MAP = {
    "segformer_b0": "nvidia/mit-b0",
    "segformer_b1": "nvidia/mit-b1",
    "segformer_b2": "nvidia/mit-b2",
    "segformer_b3": "nvidia/mit-b3",
    "segformer_b4": "nvidia/mit-b4",
    "segformer_b5": "nvidia/mit-b5",
}


def _try_login_hf():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("No HF_TOKEN found in environment. Please login manually if required.")
        return
    try:
        login(token=hf_token)
    except Exception as exc:
        print(f"HuggingFace login failed: {exc}")


def _load_label_map(num_classes: int) -> Tuple[dict, dict]:
    try:
        id2label_path = hf_hub_download(
            repo_id="huggingface/label-files",
            filename="ade20k-id2label.json",
            repo_type="dataset",
        )
        with open(id2label_path, "r", encoding="utf-8") as handle:
            id2label = json.load(handle)
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
    except Exception as exc:
        print(f"Failed to load label mapping: {exc}")
        id2label = {i: str(i) for i in range(num_classes)}
        label2id = {str(i): i for i in range(num_classes)}
    return id2label, label2id


def resolve_hf_model_id(model_name: str) -> str:
    key = str(model_name).lower()
    return SEGFORMER_NAME_MAP.get(key, model_name)


def _normalize_checkpoint(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"none", "null"}:
            return None
        return stripped
    return str(value)


def _build_segformer_config(
    model_name: str,
    num_classes: int,
    ignore_index: int,
    pretrained: Optional[str] = None,
    pretrained_encoder: Optional[str] = None,
):
    resolved_name = resolve_hf_model_id(model_name)
    config_source = pretrained or pretrained_encoder or resolved_name

    config = SegformerConfig.from_pretrained(config_source)
    config.num_labels = num_classes
    config.ignore_index = ignore_index
    config.output_hidden_states = True

    id2label, label2id = _load_label_map(num_classes)
    config.id2label = id2label
    config.label2id = label2id
    return resolved_name, config


def _build_segformer_model(
    model_name: str,
    num_classes: int,
    pretrained: Optional[str] = None,
    pretrained_encoder: Optional[str] = None,
    ignore_index: int = 255,
):
    pretrained = _normalize_checkpoint(pretrained)
    pretrained_encoder = _normalize_checkpoint(pretrained_encoder)

    _try_login_hf()

    resolved_name, config = _build_segformer_config(
        model_name=model_name,
        num_classes=num_classes,
        ignore_index=ignore_index,
        pretrained=pretrained,
        pretrained_encoder=pretrained_encoder,
    )

    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            config=config,
            ignore_mismatched_sizes=True,
        )

    if pretrained_encoder:
        model = SegformerForSemanticSegmentation(config)
        encoder = SegformerModel.from_pretrained(pretrained_encoder)
        model.segformer.load_state_dict(encoder.state_dict(), strict=False)
        return model

    return SegformerForSemanticSegmentation.from_pretrained(
        resolved_name,
        config=config,
        ignore_mismatched_sizes=True,
    )


class SegformerWrapper(torch.nn.Module):
    def __init__(self, name: str, num_classes: int, model: SegformerForSemanticSegmentation):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.model = model

    def extract_feature(self, x):
        outputs = self.model(x, output_hidden_states=True, return_dict=True)
        feats = list(outputs.hidden_states)
        attens = list(outputs.hidden_states)
        logits = outputs.logits
        return feats, attens, logits

    def forward(self, x):
        return self.model(x).logits


def build_segformer(
    model_name: str,
    num_classes: int,
    pretrained: Optional[str] = None,
    pretrained_encoder: Optional[str] = None,
    ignore_index: int = 255,
) -> SegformerWrapper:
    if not model_name:
        raise ValueError("model_name is required")

    model = _build_segformer_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_encoder=pretrained_encoder,
        ignore_index=ignore_index,
    )
    return SegformerWrapper(model_name, num_classes, model)
