import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller


def build_feature_connector(t_channel, s_channel):
    C = [
        nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(t_channel),
    ]
    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return nn.Sequential(*C)


class AttnFD(Distiller):
    """Attention-guided Feature Distillation (AttnFD, Eq. 6–7 in paper)."""

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        distill = cfg["distill"]
        self.ce_loss_weight = distill.get("ce_weight", 1.0)
        # alpha in L_AttnFD = L_CE + alpha * L_Attn (paper Sec. 3.3)
        self.attn_lambda = distill.get("attn_lambda", 2.0)
        self.attn_start_layer = distill.get("attn_start_layer", 0)

        img_size = cfg["dataset"]["img_size"]
        h, w = (img_size[0], img_size[1]) if isinstance(img_size, (list, tuple)) else (img_size, img_size)
        device = next(self.student.parameters()).device
        dummy_input = torch.randn(2, 3, h, w, device=device)

        self.student.eval()
        self.teacher.eval()
        with torch.no_grad():
            s_feats, s_attens, _ = self.student.extract_feature(dummy_input)
            t_feats, t_attens, _ = self.teacher.extract_feature(dummy_input)

        if len(s_attens) != len(t_attens):
            raise ValueError(
                f"Student/teacher attention stages mismatch: {len(s_attens)} vs {len(t_attens)}"
            )

        self.Connectors = nn.ModuleList()
        for i in range(len(t_attens)):
            t_c = t_attens[i].shape[1]
            s_c = s_attens[i].shape[1]
            self.Connectors.append(build_feature_connector(t_c, s_c))
        self.Connectors.to(device)

    def get_connector_parameters(self):
        return list(self.Connectors.parameters())

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + self.get_connector_parameters()

    def get_extra_parameters(self):
        return sum(p.numel() for p in self.Connectors.parameters())

    def _attnfd_loss(self, s_attens, t_attens):
        """L2-normalized MSE over attention maps (Eq. 6), summed over selected stages."""
        loss_attnfd = 0.0
        count = 0
        for i in range(self.attn_start_layer, len(s_attens)):
            s_attn = self.Connectors[i](s_attens[i])
            t_attn = t_attens[i]
            if s_attn.shape[2:] != t_attn.shape[2:]:
                s_attn = F.interpolate(
                    s_attn, size=t_attn.shape[2:], mode="bilinear", align_corners=False
                )
            b = s_attn.shape[0]
            s_norm = s_attn / (s_attn.norm(p=2) + 1e-6)
            t_norm = t_attn / (t_attn.norm(p=2) + 1e-6)
            loss_attnfd += (s_norm - t_norm).pow(2).sum() / b
            count += 1
        if count == 0:
            return s_attens[0].sum() * 0.0
        return loss_attnfd * self.attn_lambda

    def forward_train(self, image, target, **kwargs):
        s_feats, s_attens, logits_student = self.student.extract_feature(image)
        with torch.no_grad():
            _, t_attens, _ = self.teacher.extract_feature(image)

        logits_student = F.interpolate(
            logits_student, size=target.shape[1:], mode="bilinear", align_corners=False
        )
        loss_ce = self.ce_loss_weight * F.cross_entropy(
            logits_student, target, ignore_index=255
        )
        loss_attnfd = self._attnfd_loss(s_attens, t_attens)

        return logits_student, {
            "loss_ce": loss_ce,
            "loss_kd": loss_attnfd,
            "loss_total": loss_ce + loss_attnfd,
        }
