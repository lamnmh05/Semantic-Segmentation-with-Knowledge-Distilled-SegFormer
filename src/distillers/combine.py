import torch
import torch.nn.functional as F

from .base import Distiller
from .mlp import MLPTransform


class Combine(Distiller):
    """Combined MLP feature distillation + BPKD for semantic segmentation."""

    @staticmethod
    def _to_scalar(value, name):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be a scalar value, got tensor with shape {tuple(value.shape)}")
            return float(value.item())

        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"{name} must be a scalar value, got {value!r}")
            value = value[0]

        return float(value)

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)

        distill = cfg["distill"]

        self.ce_loss_weight = self._to_scalar(
            distill.get("lambda_ce", distill.get("ce_weight", 1.0)),
            "lambda_ce",
        )
        self.mlp_loss_weight = self._to_scalar(
            distill.get("lambda_mlp", distill.get("feat_weight", 1.0)),
            "lambda_mlp",
        )
        self.bpkd_loss_weight = self._to_scalar(
            distill.get("lambda_bpkd", distill.get("kd_weight", 1.0)),
            "lambda_bpkd",
        )

        self.lambda_body = self._to_scalar(distill.get("lambda_body", 20.0), "lambda_body")
        self.lambda_edge = self._to_scalar(distill.get("lambda_edge", 50.0), "lambda_edge")
        self.alpha_edge = self._to_scalar(distill.get("alpha_edge", 2.0), "alpha_edge")
        self.temperature = self._to_scalar(distill.get("temperature", 4.0), "temperature")
        self.ignore_index = int(distill.get("ignore_index", 255))

        edge_width = distill.get("edge_width", None)
        if edge_width is None:
            self.boundary_radius = int(
                self._to_scalar(distill.get("boundary_radius", 3), "boundary_radius")
            )
            self.edge_width = 2 * self.boundary_radius + 1
        else:
            self.edge_width = int(self._to_scalar(edge_width, "edge_width"))
            if self.edge_width % 2 == 0:
                raise ValueError("edge_width must be odd, e.g. 3, 5, 7, 9.")
            self.boundary_radius = self.edge_width // 2

        self.feat_layer = distill.get("feat_layer", 2)

        student_channels = distill.get("student_channels", None)
        teacher_channels = distill.get("teacher_channels", None)

        if student_channels is None:
            raise ValueError("Missing cfg['distill']['student_channels'].")

        if teacher_channels is None:
            raise ValueError("Missing cfg['distill']['teacher_channels'].")

        if self.feat_layer >= len(student_channels):
            raise ValueError(
                f"feat_layer={self.feat_layer} out of range for student_channels "
                f"with length {len(student_channels)}"
            )

        if self.feat_layer >= len(teacher_channels):
            raise ValueError(
                f"feat_layer={self.feat_layer} out of range for teacher_channels "
                f"with length {len(teacher_channels)}"
            )

        s_channel = student_channels[self.feat_layer]
        t_channel = teacher_channels[self.feat_layer]
        hidden_channel = distill.get("mlp_hidden_channel", t_channel)

        self.mlp = MLPTransform(
            s_channel=s_channel,
            t_channel=t_channel,
            hidden_channel=hidden_channel,
        )

        device = next(self.student.parameters()).device
        self.mlp.to(device)

    def get_mlp_parameters(self):
        return list(self.mlp.parameters())

    def get_regressor_parameters(self):
        return self.get_mlp_parameters()

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + self.get_mlp_parameters()

    def get_extra_parameters(self):
        return sum(p.numel() for p in self.mlp.parameters())

    def _resize_logits(self, logits, size):
        if logits.shape[-2:] != size:
            logits = F.interpolate(
                logits,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def _pool_to_logits(self, x, logits_size):
        H, W = x.shape[-2:]
        H_logit, W_logit = logits_size

        if H == H_logit and W == W_logit:
            return x

        if H % H_logit == 0 and W % W_logit == 0:
            stride_h = H // H_logit
            stride_w = W // W_logit

            return F.avg_pool2d(
                x,
                kernel_size=(stride_h, stride_w),
                stride=(stride_h, stride_w),
            )

        return F.adaptive_avg_pool2d(x, output_size=logits_size)

    def _prepare_target(self, target):
        if target.dim() == 4:
            target = target.squeeze(1)

        if target.dim() != 3:
            raise ValueError(
                f"target must have shape [B, H, W] or [B, 1, H, W], got {target.shape}"
            )

        return target.long()

    def feature_loss(self, s_feats, t_feats):
        idx = int(self.feat_layer)
        
        f_s = s_feats[idx]
        f_t = t_feats[idx]

        f_s_proj = self.mlp(f_s)

        if f_s_proj.shape[2:] != f_t.shape[2:]:
            f_s_proj = F.interpolate(f_s_proj, size=f_t.shape[2:], mode="bilinear", align_corners=False)

        diff = (f_s_proj - f_t).pow(2)
        return diff.sum() / diff.size(0)

    def get_edge_mask(self, target, num_classes, logits_size):
        target = self._prepare_target(target)

        B, H, W = target.shape

        valid_mask = target != self.ignore_index

        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        gt_onehot = F.one_hot(
            target_safe,
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).float()

        valid_mask_float = valid_mask.unsqueeze(1).float()
        gt_onehot = gt_onehot * valid_mask_float

        kernel_size = self.edge_width
        padding = self.boundary_radius

        dilation = F.max_pool2d(
            gt_onehot,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        erosion = 1.0 - F.max_pool2d(
            1.0 - gt_onehot,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        gt_edge = (dilation - erosion).clamp(min=0.0, max=1.0)
        gt_edge = gt_edge * valid_mask_float

        edge_mask = self._pool_to_logits(gt_edge, logits_size).clamp(min=0.0, max=1.0)
        valid_mask_logits = self._pool_to_logits(valid_mask_float, logits_size).clamp(
            min=0.0, max=1.0
        )

        edge_mask = edge_mask * valid_mask_logits

        return edge_mask, valid_mask_logits

    def edge_loss(self, logits_student, logits_teacher, edge_mask):
        T = self.temperature

        B, C, H, W = logits_student.shape

        edge_mask = edge_mask.to(
            device=logits_student.device,
            dtype=logits_student.dtype,
        )

        if edge_mask.shape[-2:] != (H, W):
            edge_mask = F.interpolate(
                edge_mask,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        z_s_edge = logits_student * edge_mask
        z_t_edge = logits_teacher.detach() * edge_mask

        log_prob_s = F.log_softmax(z_s_edge / T, dim=1)
        prob_t = F.softmax(z_t_edge / T, dim=1)

        kl_map_class = F.kl_div(
            log_prob_s,
            prob_t,
            reduction="none",
        ) * (T * T)

        phi_pixel = kl_map_class.sum(dim=1, keepdim=True)
        phi_repeat = phi_pixel.expand(-1, C, -1, -1)

        n_c = (edge_mask > 0).float().sum(dim=(2, 3)).clamp_min(1.0)
        loss_per_class = (phi_repeat * edge_mask).sum(dim=(2, 3)) / n_c
        loss_edge = loss_per_class.sum(dim=1).mean()

        return self.alpha_edge * loss_edge

    def body_loss(self, logits_student, logits_teacher, edge_mask, valid_mask_logits):
        T = self.temperature

        B, C, H, W = logits_student.shape

        edge_mask = edge_mask.to(
            device=logits_student.device,
            dtype=logits_student.dtype,
        )

        valid_mask_logits = valid_mask_logits.to(
            device=logits_student.device,
            dtype=logits_student.dtype,
        )

        if edge_mask.shape[-2:] != (H, W):
            edge_mask = F.interpolate(
                edge_mask,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        if valid_mask_logits.shape[-2:] != (H, W):
            valid_mask_logits = F.interpolate(
                valid_mask_logits,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        body_mask = (1.0 - edge_mask) * valid_mask_logits

        z_s_body = logits_student * body_mask
        z_t_body = logits_teacher.detach() * body_mask

        z_s_body = z_s_body.flatten(2)
        z_t_body = z_t_body.flatten(2)

        log_prob_s = F.log_softmax(z_s_body / T, dim=2)
        prob_t = F.softmax(z_t_body / T, dim=2)

        kl_map = F.kl_div(
            log_prob_s,
            prob_t,
            reduction="none",
        )

        return (T * T) * kl_map.sum(dim=2).sum(dim=1).mean() / C

    def bpkd_loss(self, logits_student, logits_teacher, target):
        if logits_student.dim() != 4:
            raise ValueError(
                f"logits_student must have shape [B, C, H, W], got {logits_student.shape}"
            )

        if logits_teacher.dim() != 4:
            raise ValueError(
                f"logits_teacher must have shape [B, C, H, W], got {logits_teacher.shape}"
            )

        logits_size = logits_student.shape[-2:]
        logits_teacher = self._resize_logits(logits_teacher, logits_size)
        num_classes = logits_student.shape[1]

        edge_mask, valid_mask_logits = self.get_edge_mask(
            target=target,
            num_classes=num_classes,
            logits_size=logits_size,
        )

        loss_edge = self.edge_loss(
            logits_student=logits_student,
            logits_teacher=logits_teacher,
            edge_mask=edge_mask,
        )
        loss_body = self.body_loss(
            logits_student=logits_student,
            logits_teacher=logits_teacher,
            edge_mask=edge_mask,
            valid_mask_logits=valid_mask_logits,
        )

        loss_body_w = self.lambda_body * loss_body
        loss_edge_w = self.lambda_edge * loss_edge
        loss_bpkd = loss_body_w + loss_edge_w

        return {
            "loss_bpkd": loss_bpkd,
            "loss_body": loss_body,
            "loss_edge": loss_edge,
            "loss_body_w": loss_body_w,
            "loss_edge_w": loss_edge_w,
        }

    def forward_train(self, image, target, **kwargs):
        target = self._prepare_target(target)

        s_feats, _, logits_student = self.student.extract_feature(image)

        with torch.no_grad():
            t_feats, _, logits_teacher = self.teacher.extract_feature(image)

        logits_student_for_ce = self._resize_logits(
            logits_student,
            target.shape[-2:],
        )

        loss_ce = self.ce_loss_weight * F.cross_entropy(
            logits_student_for_ce,
            target,
            ignore_index=self.ignore_index,
        )

        loss_mlp_raw = self.feature_loss(s_feats, t_feats)
        loss_mlp = self.mlp_loss_weight * loss_mlp_raw

        loss_dict = self.bpkd_loss(
            logits_student=logits_student,
            logits_teacher=logits_teacher,
            target=target,
        )

        loss_bpkd = self.bpkd_loss_weight * loss_dict["loss_bpkd"]

        loss_kd = loss_mlp + loss_bpkd
        loss_total = loss_ce + loss_kd

        return logits_student_for_ce, {
            "loss_ce": loss_ce,
            "loss_mlp": loss_mlp,
            "loss_body": loss_dict["loss_body"],
            "loss_edge": loss_dict["loss_edge"],
            "loss_body_w": loss_dict["loss_body_w"],
            "loss_edge_w": loss_dict["loss_edge_w"],
            "loss_bpkd": loss_bpkd,
            "loss_kd": loss_kd,
            "loss_total": loss_total,
        }