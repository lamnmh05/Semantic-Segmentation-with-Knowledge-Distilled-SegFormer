import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller


class MLPTransform(nn.Module):
    """
    Channel-wise transformation using MLP.
        student feature: [B, C_s, H, W]
        teacher feature: [B, C_t, H, W]

        [B, C_s, H, W] -> [B, C_t, H, W]
    """

    def __init__(self, s_channel, t_channel, hidden_channel=None):
        super().__init__()

        if hidden_channel is None:
            hidden_channel = t_channel

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=s_channel,
                out_channels=hidden_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channel,
                out_channels=t_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x):
        return self.net(x)


class MLPFD(Distiller):
    """
    MLP-based Feature Distillation
    Loss:
        loss_total = ce_weight * CE(logits_student, target)
                   + feat_weight * MSE(MLP(f_s), f_t)
    """

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)

        distill = cfg["distill"]

        self.ce_loss_weight = float(distill.get("ce_weight", 1.0))
        self.feat_loss_weight = float(distill.get("feat_weight", 1.0))
        self.ignore_index = int(distill.get("ignore_index", 255))


        self.feat_layer = int(distill.get("feat_layer", 2))

        # Không dùng dummy input.
        # Vì vậy cần khai báo channel trong config.
        student_channels = distill.get("student_channels", None)
        teacher_channels = distill.get("teacher_channels", None)

        if student_channels is None:
            raise ValueError(
                "Missing cfg['distill']['student_channels']."
            )

        if teacher_channels is None:
            raise ValueError(
                "Missing cfg['distill']['teacher_channels']. "
            )

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

        # Nếu student đã nằm trên GPU trước khi tạo distiller,
        # đưa MLP sang cùng device.
        device = next(self.student.parameters()).device
        self.mlp.to(device)

    def get_mlp_parameters(self):
        return list(self.mlp.parameters())

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + self.get_mlp_parameters()

    def get_extra_parameters(self):
        return sum(p.numel() for p in self.mlp.parameters())

    def feature_loss(self, s_feats, t_feats):
        """
        Tính feature distillation loss:

            MSE(MLP(f_s), f_t)
        """

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

        loss = F.mse_loss(f_s, f_t)

        return loss

    def forward_train(self, image, target, **kwargs):
        s_feats, _, logits_student = self.student.extract_feature(image)

        with torch.no_grad():
            t_feats, _, _ = self.teacher.extract_feature(image)

        loss_feat = self.feat_loss_weight * self.feature_loss(s_feats, t_feats)

        logits_student = F.interpolate(
            logits_student,
            size=target.shape[1:],
            mode="bilinear",
            align_corners=False,
        )

        loss_ce = self.ce_loss_weight * F.cross_entropy(
            logits_student,
            target,
            ignore_index=self.ignore_index,
        )

        loss_total = loss_ce + loss_feat

        return logits_student, {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
            "loss_total": loss_total,
        }