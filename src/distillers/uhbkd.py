import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller
from .mlp import MLPTransform


class UHBKD(Distiller):
    """Uncertainty-guided Hierarchical Boundary-aware Distillation.

    Components
    ----------
    1. CE loss with ground-truth labels.
    2. Hierarchical feature distillation on multiple stages [0, 1, 2, 3].
    3. Boundary-aware KD: separate body and edge regions generated from GT masks.
    4. Teacher-confidence weighting: strong KD where the teacher is confident, softer KD
       where the teacher is uncertain.
    5. Optional class-wise boundary adaptation with learnable per-class edge weights.

    Expected model API
    ------------------
    student.extract_feature(image) -> (student_features, _, student_logits)
    teacher.extract_feature(image) -> (teacher_features, _, teacher_logits)

    student_features and teacher_features should be lists/tuples of feature maps:
        stage i: [B, C_i, H_i, W_i]
    """

    @staticmethod
    def _to_scalar(value, name):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be scalar, got tensor with shape {tuple(value.shape)}")
            return float(value.item())

        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"{name} must be scalar, got {value!r}")
            value = value[0]

        return float(value)

    @staticmethod
    def _as_list(value, name, length=None, dtype=None):
        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            out = list(value)
        else:
            if length is None:
                out = [value]
            else:
                out = [value for _ in range(length)]

        if length is not None and len(out) != length:
            raise ValueError(f"{name} must have length {length}, got length {len(out)}: {out!r}")

        if dtype is not None:
            out = [dtype(x) for x in out]

        return out

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)

        distill = cfg["distill"]

        # ------------------------------------------------------------------
        # Main loss weights
        # ------------------------------------------------------------------
        self.ce_loss_weight = self._to_scalar(
            distill.get("lambda_ce", distill.get("ce_weight", 1.0)),
            "lambda_ce",
        )
        self.feat_loss_weight = self._to_scalar(
            distill.get("lambda_feat", distill.get("lambda_mlp", distill.get("feat_weight", 1.0))),
            "lambda_feat",
        )
        self.bpkd_loss_weight = self._to_scalar(
            distill.get("lambda_bpkd", distill.get("kd_weight", 1.0)),
            "lambda_bpkd",
        )
        self.uncertainty_loss_weight = self._to_scalar(
            distill.get("lambda_uncertainty", 0.0),
            "lambda_uncertainty",
        )

        # ------------------------------------------------------------------
        # BPKD weights
        # ------------------------------------------------------------------
        self.lambda_body = self._to_scalar(distill.get("lambda_body", 2.0), "lambda_body")
        self.lambda_edge = self._to_scalar(distill.get("lambda_edge", 4.0), "lambda_edge")
        self.alpha_edge = self._to_scalar(distill.get("alpha_edge", 2.0), "alpha_edge")
        self.temperature = self._to_scalar(distill.get("temperature", 2.0), "temperature")
        self.ignore_index = int(distill.get("ignore_index", 255))

        edge_width = int(self._to_scalar(distill.get("edge_width", 7), "edge_width"))
        if edge_width % 2 == 0:
            raise ValueError("edge_width must be odd, e.g. 3, 5, 7, 9.")
        self.edge_width = edge_width
        self.boundary_radius = edge_width // 2

        # edge_loss_mode='class': use class-wise edge mask [B, C, H, W]
        # edge_loss_mode='union': use union edge mask [B, 1, H, W]
        self.edge_loss_mode = str(distill.get("edge_loss_mode", "class")).lower()
        if self.edge_loss_mode not in {"class", "union"}:
            raise ValueError("edge_loss_mode must be either 'class' or 'union'.")

        # Uncertainty / confidence settings
        self.uncertainty_type = str(distill.get("uncertainty_type", "entropy")).lower()
        if self.uncertainty_type not in {"entropy", "margin"}:
            raise ValueError("uncertainty_type must be either 'entropy' or 'margin'.")

        self.min_teacher_confidence = self._to_scalar(
            distill.get("min_teacher_confidence", 0.05),
            "min_teacher_confidence",
        )
        self.confidence_power = self._to_scalar(
            distill.get("confidence_power", 1.0),
            "confidence_power",
        )
        self.detach_teacher_confidence = bool(distill.get("detach_teacher_confidence", True))

        # Hierarchical feature distillation settings
        student_channels = distill.get("student_channels", None)
        teacher_channels = distill.get("teacher_channels", None)

        if student_channels is None:
            raise ValueError("Missing cfg['distill']['student_channels'].")
        if teacher_channels is None:
            raise ValueError("Missing cfg['distill']['teacher_channels'].")

        student_channels = self._as_list(student_channels, "student_channels", dtype=int)
        teacher_channels = self._as_list(teacher_channels, "teacher_channels", dtype=int)

        if len(student_channels) != len(teacher_channels):
            raise ValueError(
                "student_channels and teacher_channels must have the same length, "
                f"got {len(student_channels)} and {len(teacher_channels)}."
            )

        feat_layers = distill.get("feat_layers", None)
        if feat_layers is None:
            # Backward-compatible fallback: old config used feat_layer: 3.
            # For UH-BBKD, default to all available stages.
            if "feat_layer" in distill:
                feat_layers = [int(distill["feat_layer"])]
            else:
                feat_layers = list(range(len(student_channels)))

        self.feat_layers = self._as_list(feat_layers, "feat_layers", dtype=int)
        self.num_stages = len(self.feat_layers)

        for idx in self.feat_layers:
            if idx < 0 or idx >= len(student_channels):
                raise ValueError(
                    f"feat layer index {idx} out of range for channel list length {len(student_channels)}."
                )

        hidden_channels = distill.get("mlp_hidden_channels", distill.get("mlp_hidden_channel", None))
        if hidden_channels is None:
            hidden_channels = [teacher_channels[i] for i in self.feat_layers]
        hidden_channels = self._as_list(hidden_channels, "mlp_hidden_channels", length=self.num_stages, dtype=int)

        stage_weights = distill.get("stage_weights", [1.0 for _ in range(self.num_stages)])
        stage_weights = self._as_list(stage_weights, "stage_weights", length=self.num_stages, dtype=float)
        stage_weights = torch.tensor(stage_weights, dtype=torch.float32)
        stage_weights = stage_weights / stage_weights.sum().clamp_min(1e-12)
        self.register_buffer("stage_weights", stage_weights)

        self.stage_mlps = nn.ModuleList()
        for layer_idx, hidden_ch in zip(self.feat_layers, hidden_channels):
            self.stage_mlps.append(
                MLPTransform(
                    s_channel=student_channels[layer_idx],
                    t_channel=teacher_channels[layer_idx],
                    hidden_channel=hidden_ch,
                )
            )

        # Boundary-aware weighting inside hierarchical feature loss.
        self.feature_boundary_weight = self._to_scalar(
            distill.get("feature_boundary_weight", 1.0),
            "feature_boundary_weight",
        )
        self.feature_use_confidence = bool(distill.get("feature_use_confidence", True))

        # Optional class-wise boundary adaptation.
        self.use_classwise_boundary = bool(distill.get("use_classwise_boundary", True))
        num_classes = int(cfg.get("model", {}).get("num_classes", distill.get("num_classes", 150)))
        self.num_classes = num_classes

        if self.use_classwise_boundary:
            initial_weight = self._to_scalar(
                distill.get("boundary_initial_weight", 1.0),
                "boundary_initial_weight",
            )
            self.class_boundary_weight = nn.Parameter(torch.ones(num_classes) * initial_weight)
        else:
            self.class_boundary_weight = None

        # Put connector modules on the same device as the student.
        device = next(self.student.parameters()).device
        self.stage_mlps.to(device)
        if self.class_boundary_weight is not None:
            self.class_boundary_weight.data = self.class_boundary_weight.data.to(device)

    # Parameter helpers for compatibility with existing training code
    def get_connector_parameters(self):
        params = list(self.stage_mlps.parameters())
        if self.class_boundary_weight is not None:
            params.append(self.class_boundary_weight)
        return params

    def get_mlp_parameters(self):
        return self.get_connector_parameters()

    def get_regressor_parameters(self):
        return self.get_connector_parameters()

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + self.get_connector_parameters()

    def get_extra_parameters(self):
        total = sum(p.numel() for p in self.stage_mlps.parameters())
        if self.class_boundary_weight is not None:
            total += self.class_boundary_weight.numel()
        return total

    # Tensor utilities
    def _prepare_target(self, target):
        if target.dim() == 4:
            target = target.squeeze(1)

        if target.dim() != 3:
            raise ValueError(
                f"target must have shape [B, H, W] or [B, 1, H, W], got {tuple(target.shape)}"
            )

        return target.long()

    @staticmethod
    def _resize_logits(logits, size):
        if logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return logits

    def _resize_target(self, target, size):
        target = self._prepare_target(target)
        if target.shape[-2:] != size:
            target = F.interpolate(target.unsqueeze(1).float(), size=size, mode="nearest").squeeze(1).long()
        return target

    @staticmethod
    def _weighted_mean(value, weight, eps=1e-6):
        """Weighted mean for value [B, 1 or C, H, W] and broadcastable weight."""
        weight = weight.to(device=value.device, dtype=value.dtype)
        while weight.dim() < value.dim():
            weight = weight.unsqueeze(1)

        weighted = value * weight
        denom = weight.expand_as(value).sum().clamp_min(eps)
        return weighted.sum() / denom

    # Uncertainty / confidence
    def _confidence_from_logits(self, logits):
        """Return (confidence, uncertainty), both shaped [B, 1, H, W]."""
        T = self.temperature
        probs = F.softmax(logits / T, dim=1)
        B, C, H, W = probs.shape

        if self.uncertainty_type == "entropy":
            entropy = -(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=1, keepdim=True)
            max_entropy = torch.log(torch.tensor(float(C), device=logits.device, dtype=logits.dtype))
            uncertainty = (entropy / max_entropy.clamp_min(1e-10)).clamp(0.0, 1.0)
            confidence = 1.0 - uncertainty
        else:
            top2 = torch.topk(probs, k=2, dim=1).values
            confidence = (top2[:, 0:1] - top2[:, 1:2]).clamp(0.0, 1.0)
            uncertainty = 1.0 - confidence

        confidence = confidence.clamp(0.0, 1.0)
        if self.confidence_power != 1.0:
            confidence = confidence.pow(self.confidence_power)

        # Keep a small non-zero KD signal even in uncertain areas.
        confidence = self.min_teacher_confidence + (1.0 - self.min_teacher_confidence) * confidence
        confidence = confidence.clamp(0.0, 1.0)

        return confidence, uncertainty

    def _student_teacher_uncertainty_loss(self, logits_student, logits_teacher, valid_mask):
        if self.uncertainty_loss_weight <= 0:
            return logits_student.new_tensor(0.0)

        student_conf, _ = self._confidence_from_logits(logits_student)
        with torch.no_grad():
            teacher_conf, _ = self._confidence_from_logits(logits_teacher)

        loss_map = (student_conf - teacher_conf.detach()).pow(2)
        return self._weighted_mean(loss_map, valid_mask)

    # Boundary mask generation
    def get_boundary_masks(self, target, num_classes, size):
        """Generate class-wise and union boundary masks at a given resolution.

        Returns
        -------
        edge_class: [B, C, H, W]
        edge_union: [B, 1, H, W]
        valid_mask: [B, 1, H, W]
        """
        target = self._resize_target(target, size)
        valid_mask_bool = target != self.ignore_index

        target_safe = target.clone()
        target_safe[~valid_mask_bool] = 0

        gt_onehot = F.one_hot(target_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask_bool.unsqueeze(1).float()
        gt_onehot = gt_onehot * valid_mask

        dilation = F.max_pool2d(
            gt_onehot,
            kernel_size=self.edge_width,
            stride=1,
            padding=self.boundary_radius,
        )
        erosion = 1.0 - F.max_pool2d(
            1.0 - gt_onehot,
            kernel_size=self.edge_width,
            stride=1,
            padding=self.boundary_radius,
        )

        edge_class = (dilation - erosion).clamp(0.0, 1.0) * valid_mask
        edge_union = edge_class.max(dim=1, keepdim=True).values

        return edge_class, edge_union, valid_mask

    def _classwise_edge_weight(self, edge_class):
        if self.class_boundary_weight is None:
            return edge_class

        # Use softplus to keep weights positive even after optimization.
        weights = F.softplus(self.class_boundary_weight).view(1, -1, 1, 1)
        return edge_class * weights.to(device=edge_class.device, dtype=edge_class.dtype)

    # Hierarchical feature distillation
    def hierarchical_feature_loss(self, s_feats, t_feats, target, teacher_confidence):
        if len(s_feats) <= max(self.feat_layers):
            raise ValueError(
                f"student feature list has length {len(s_feats)}, but feat_layers={self.feat_layers}."
            )
        if len(t_feats) <= max(self.feat_layers):
            raise ValueError(
                f"teacher feature list has length {len(t_feats)}, but feat_layers={self.feat_layers}."
            )

        total_loss = s_feats[0].new_tensor(0.0)
        loss_items = {}

        for local_idx, feat_idx in enumerate(self.feat_layers):
            f_s = s_feats[feat_idx]
            f_t = t_feats[feat_idx].detach()

            f_s_proj = self.stage_mlps[local_idx](f_s)

            if f_s_proj.shape[-2:] != f_t.shape[-2:]:
                f_s_proj = F.interpolate(f_s_proj, size=f_t.shape[-2:], mode="bilinear", align_corners=False)

            diff = (f_s_proj - f_t).pow(2)

            _, edge_union, valid_mask = self.get_boundary_masks(
                target=target,
                num_classes=self.num_classes,
                size=f_t.shape[-2:],
            )
            edge_union = edge_union.to(device=diff.device, dtype=diff.dtype)
            valid_mask = valid_mask.to(device=diff.device, dtype=diff.dtype)

            weight = valid_mask * (1.0 + self.feature_boundary_weight * edge_union)

            if self.feature_use_confidence:
                conf_stage = F.interpolate(
                    teacher_confidence,
                    size=f_t.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                weight = weight * conf_stage.to(device=diff.device, dtype=diff.dtype)

            stage_loss = self._weighted_mean(diff, weight)
            stage_loss = self.stage_weights[local_idx].to(stage_loss.device) * stage_loss
            total_loss = total_loss + stage_loss

            loss_items[f"loss_feat_s{feat_idx}"] = stage_loss.detach()

        return total_loss, loss_items

    # Boundary-aware KD
    def bpkd_loss(self, logits_student, logits_teacher, target, teacher_confidence):
        if logits_student.dim() != 4:
            raise ValueError(f"logits_student must be [B, C, H, W], got {tuple(logits_student.shape)}")
        if logits_teacher.dim() != 4:
            raise ValueError(f"logits_teacher must be [B, C, H, W], got {tuple(logits_teacher.shape)}")

        B, C, H, W = logits_student.shape
        logits_teacher = self._resize_logits(logits_teacher, (H, W)).detach()

        edge_class, edge_union, valid_mask = self.get_boundary_masks(
            target=target,
            num_classes=C,
            size=(H, W),
        )
        edge_class = edge_class.to(device=logits_student.device, dtype=logits_student.dtype)
        edge_union = edge_union.to(device=logits_student.device, dtype=logits_student.dtype)
        valid_mask = valid_mask.to(device=logits_student.device, dtype=logits_student.dtype)

        confidence = teacher_confidence.to(device=logits_student.device, dtype=logits_student.dtype)
        if confidence.shape[-2:] != (H, W):
            confidence = F.interpolate(confidence, size=(H, W), mode="bilinear", align_corners=False)
        if self.detach_teacher_confidence:
            confidence = confidence.detach()

        T = self.temperature
        log_prob_s = F.log_softmax(logits_student / T, dim=1)
        prob_t = F.softmax(logits_teacher / T, dim=1)

        # Class-wise KL map: [B, C, H, W]
        kl_class = F.kl_div(log_prob_s, prob_t, reduction="none") * (T * T)

        # Body region: use union mask because body is the non-boundary part.
        body_mask = (1.0 - edge_union) * valid_mask * confidence
        kl_pixel = kl_class.sum(dim=1, keepdim=True)
        loss_body = self._weighted_mean(kl_pixel, body_mask)

        # Edge region: either class-specific boundary or union boundary.
        if self.edge_loss_mode == "class":
            edge_weight = self._classwise_edge_weight(edge_class)
            edge_weight = edge_weight * valid_mask * confidence
            loss_edge = self._weighted_mean(kl_class, edge_weight)
        else:
            edge_weight = edge_union * valid_mask * confidence
            loss_edge = self._weighted_mean(kl_pixel, edge_weight)

        loss_edge = self.alpha_edge * loss_edge

        loss_body_w = self.lambda_body * loss_body
        loss_edge_w = self.lambda_edge * loss_edge
        loss_bpkd = loss_body_w + loss_edge_w

        return {
            "loss_bpkd_raw": loss_bpkd,
            "loss_body": loss_body,
            "loss_edge": loss_edge,
            "loss_body_w": loss_body_w,
            "loss_edge_w": loss_edge_w,
            "edge_ratio": edge_union.mean().detach(),
            "valid_ratio": valid_mask.mean().detach(),
        }

    # Main forward
    def forward_train(self, image, target, **kwargs):
        target = self._prepare_target(target)

        s_feats, _, logits_student = self.student.extract_feature(image)

        with torch.no_grad():
            t_feats, _, logits_teacher = self.teacher.extract_feature(image)

        # CE loss at GT resolution.
        logits_student_for_ce = self._resize_logits(logits_student, target.shape[-2:])
        loss_ce_raw = F.cross_entropy(
            logits_student_for_ce,
            target,
            ignore_index=self.ignore_index,
        )
        loss_ce = self.ce_loss_weight * loss_ce_raw

        # Prepare teacher logits and teacher confidence at student-logit resolution.
        logits_size = logits_student.shape[-2:]
        logits_teacher_for_kd = self._resize_logits(logits_teacher, logits_size)
        teacher_confidence, teacher_uncertainty = self._confidence_from_logits(logits_teacher_for_kd.detach())

        # Hierarchical feature distillation.
        loss_feat_raw, feat_log_items = self.hierarchical_feature_loss(
            s_feats=s_feats,
            t_feats=t_feats,
            target=target,
            teacher_confidence=teacher_confidence,
        )
        loss_feat = self.feat_loss_weight * loss_feat_raw

        # Boundary-aware KD.
        bpkd_items = self.bpkd_loss(
            logits_student=logits_student,
            logits_teacher=logits_teacher_for_kd,
            target=target,
            teacher_confidence=teacher_confidence,
        )
        loss_bpkd = self.bpkd_loss_weight * bpkd_items["loss_bpkd_raw"]

        # Optional student-teacher confidence consistency.
        _, _, valid_mask_logits = self.get_boundary_masks(
            target=target,
            num_classes=logits_student.shape[1],
            size=logits_size,
        )
        valid_mask_logits = valid_mask_logits.to(device=logits_student.device, dtype=logits_student.dtype)
        loss_uncertainty_raw = self._student_teacher_uncertainty_loss(
            logits_student=logits_student,
            logits_teacher=logits_teacher_for_kd,
            valid_mask=valid_mask_logits,
        )
        loss_uncertainty = self.uncertainty_loss_weight * loss_uncertainty_raw

        loss_kd = loss_feat + loss_bpkd + loss_uncertainty
        loss_total = loss_ce + loss_kd

        log_dict = {
            "loss_ce": loss_ce,
            "loss_ce_raw": loss_ce_raw.detach(),
            "loss_feat": loss_feat,
            "loss_feat_raw": loss_feat_raw.detach(),
            "loss_body": bpkd_items["loss_body"].detach(),
            "loss_edge": bpkd_items["loss_edge"].detach(),
            "loss_body_w": bpkd_items["loss_body_w"].detach(),
            "loss_edge_w": bpkd_items["loss_edge_w"].detach(),
            "loss_bpkd": loss_bpkd,
            "loss_bpkd_raw": bpkd_items["loss_bpkd_raw"].detach(),
            "loss_uncertainty": loss_uncertainty,
            "loss_uncertainty_raw": loss_uncertainty_raw.detach(),
            "loss_kd": loss_kd,
            "loss_total": loss_total,
            "teacher_confidence_mean": teacher_confidence.mean().detach(),
            "teacher_uncertainty_mean": teacher_uncertainty.mean().detach(),
            "edge_ratio": bpkd_items["edge_ratio"],
            "valid_ratio": bpkd_items["valid_ratio"],
        }
        log_dict.update(feat_log_items)

        return logits_student_for_ce, log_dict
