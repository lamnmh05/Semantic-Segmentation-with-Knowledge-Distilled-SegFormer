import torch
import torch.nn.functional as F

from .base import Distiller
from .mlp import MLPTransform


class Combine(Distiller):
    """Combined MLP feature distillation + BPKD for semantic segmentation."""

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)

        distill = cfg["distill"]

        self.ce_loss_weight = distill.get("lambda_ce", distill.get("ce_weight", 1.0))
        self.mlp_loss_weight = distill.get("lambda_mlp", distill.get("feat_weight", 1.0))
        self.bpkd_loss_weight = distill.get("lambda_bpkd", distill.get("kd_weight", 1.0))

        self.lambda_body = distill.get("lambda_body", 20.0)
        self.lambda_edge = distill.get("lambda_edge", 50.0)
        self.alpha_edge = distill.get("alpha_edge", 2.0)
        self.temperature = distill.get("temperature", 4.0)
        self.ignore_index = distill.get("ignore_index", 255)
        self.boundary_radius = distill.get("boundary_radius", 3)

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

    def _prepare_target(self, target):
        if target.dim() == 4:
            target = target.squeeze(1)

        if target.dim() != 3:
            raise ValueError(
                f"target must have shape [B, H, W] or [B, 1, H, W], got {target.shape}"
            )

        return target.long()

    def feature_loss(self, s_feats, t_feats):
        f_s = s_feats[self.feat_layer]
        f_t = t_feats[self.feat_layer]

        f_s = self.mlp(f_s)

        if f_s.shape[2:] != f_t.shape[2:]:
            f_s = F.interpolate(
                f_s,
                size=f_t.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return F.mse_loss(f_s, f_t)

    def get_edge_mask(self, target, num_classes, logits_size):
        target = self._prepare_target(target)

        B, H, W = target.shape
        H_logit, W_logit = logits_size

        valid_mask = target != self.ignore_index

        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        gt_onehot = F.one_hot(
            target_safe,
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).float()

        gt_onehot = gt_onehot * valid_mask.unsqueeze(1).float()

        kernel_size = 2 * self.boundary_radius + 1
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
        gt_edge = gt_edge * valid_mask.unsqueeze(1).float()

        if H % H_logit == 0 and W % W_logit == 0:
            stride_h = H // H_logit
            stride_w = W // W_logit

            edge_mask = F.avg_pool2d(
                gt_edge,
                kernel_size=(stride_h, stride_w),
                stride=(stride_h, stride_w),
            )
        else:
            edge_mask = F.adaptive_avg_pool2d(
                gt_edge,
                output_size=(H_logit, W_logit),
            )

        return edge_mask.clamp(min=0.0, max=1.0)

    def edge_loss(self, logits_student, logits_teacher, edge_mask):
        T = self.temperature

        s_edge = logits_student * edge_mask
        t_edge = logits_teacher * edge_mask

        log_prob_s = F.log_softmax(s_edge / T, dim=1)
        prob_t = F.softmax(t_edge / T, dim=1)

        kl_map_class = F.kl_div(
            log_prob_s,
            prob_t,
            reduction="none",
        ) * (T * T)

        kl_map_pixel = kl_map_class.sum(dim=1, keepdim=True)
        weighted_kl = kl_map_pixel * edge_mask
        n_c = edge_mask.sum(dim=(2, 3)).clamp_min(1.0)
        loss_per_class = weighted_kl.sum(dim=(2, 3)) / n_c

        return self.alpha_edge * loss_per_class.sum(dim=1).mean()

    def body_loss(self, logits_student, logits_teacher, edge_mask):
        T = self.temperature

        B, C, H, W = logits_student.shape

        body_mask = 1.0 - edge_mask

        s_body = logits_student * body_mask
        t_body = logits_teacher * body_mask

        s_body = s_body.view(B, C, -1)
        t_body = t_body.view(B, C, -1)

        log_prob_s = F.log_softmax(s_body / T, dim=2)
        prob_t = F.softmax(t_body / T, dim=2)

        kl_per_channel = F.kl_div(
            log_prob_s,
            prob_t,
            reduction="none",
        ).sum(dim=2)

        return (T * T) * kl_per_channel.sum(dim=1).mean() / C

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

        edge_mask = self.get_edge_mask(
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
        )

        loss_bpkd = self.lambda_body * loss_body + self.lambda_edge * loss_edge
        return loss_bpkd, loss_body, loss_edge

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

        loss_bpkd_raw, loss_body_raw, loss_edge_raw = self.bpkd_loss(
            logits_student=logits_student,
            logits_teacher=logits_teacher,
            target=target,
        )
        loss_bpkd = self.bpkd_loss_weight * loss_bpkd_raw

        loss_kd = loss_mlp + loss_bpkd
        loss_total = loss_ce + loss_kd

        return logits_student_for_ce, {
            "loss_ce": loss_ce,
            "loss_mlp": loss_mlp,
            "loss_body": loss_body_raw,
            "loss_edge": loss_edge_raw,
            "loss_bpkd": loss_bpkd,
            "loss_kd": loss_kd,
            "loss_total": loss_total,
        }