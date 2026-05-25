import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller


class ConvReg(nn.Module):
    """Convolutional regressor (FitNet paper Sec. 2.2)."""

    def __init__(self, s_channel, t_channel):
        super().__init__()
        self.conv = nn.Conv2d(s_channel, t_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(t_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FitNet(Distiller):
    """FitNets hint-based feature distillation (Eq. 3) + segmentation CE."""

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        distill = cfg["distill"]
        self.ce_loss_weight = distill.get("ce_weight", 1.0)
        self.feat_loss_weight = distill.get("feat_weight", 10.0)
        # Middle encoder stage by default (paper: hint/guided at network middle).
        self.hint_layer = distill.get("hint_layer", 2)

        img_size = cfg["dataset"]["img_size"]
        h, w = (img_size[0], img_size[1]) if isinstance(img_size, (list, tuple)) else (img_size, img_size)
        device = next(self.student.parameters()).device
        dummy_input = torch.randn(2, 3, h, w, device=device)

        self.student.eval()
        self.teacher.eval()
        with torch.no_grad():
            s_feats, _, _ = self.student.extract_feature(dummy_input)
            t_feats, _, _ = self.teacher.extract_feature(dummy_input)

        if self.hint_layer >= len(s_feats) or self.hint_layer >= len(t_feats):
            raise ValueError(
                f"hint_layer={self.hint_layer} out of range for "
                f"{len(s_feats)} student / {len(t_feats)} teacher stages"
            )

        s_channel = s_feats[self.hint_layer].shape[1]
        t_channel = t_feats[self.hint_layer].shape[1]
        self.conv_reg = ConvReg(s_channel, t_channel).to(device)

    def get_regressor_parameters(self):
        return list(self.conv_reg.parameters())

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + self.get_regressor_parameters()

    def get_extra_parameters(self):
        return sum(p.numel() for p in self.conv_reg.parameters())

    def hint_loss(self, image):
        """L_HT: MSE between regressed student hint and teacher hint (Eq. 3)."""
        s_feats, _, _ = self.student.extract_feature(image)
        with torch.no_grad():
            t_feats, _, _ = self.teacher.extract_feature(image)
        f_s = self.conv_reg(s_feats[self.hint_layer])
        f_t = t_feats[self.hint_layer]
        if f_s.shape[2:] != f_t.shape[2:]:
            f_s = F.interpolate(f_s, size=f_t.shape[2:], mode="bilinear", align_corners=False)
        return F.mse_loss(f_s, f_t)

    def forward_train(self, image, target, hint_only=False, **kwargs):
        s_feats, _, logits_student = self.student.extract_feature(image)
        with torch.no_grad():
            t_feats, _, _ = self.teacher.extract_feature(image)

        f_s = self.conv_reg(s_feats[self.hint_layer])
        f_t = t_feats[self.hint_layer]
        if f_s.shape[2:] != f_t.shape[2:]:
            f_s = F.interpolate(f_s, size=f_t.shape[2:], mode="bilinear", align_corners=False)

        loss_feat = self.feat_loss_weight * F.mse_loss(f_s, f_t)

        if hint_only:
            return None, {
                "loss_ce": loss_feat.new_zeros((1,)),
                "loss_kd": loss_feat.unsqueeze(0),
                "loss_total": loss_feat.unsqueeze(0),
            }

        logits_student = F.interpolate(
            logits_student, size=target.shape[1:], mode="bilinear", align_corners=False
        )
        loss_ce = self.ce_loss_weight * F.cross_entropy(
            logits_student, target, ignore_index=255
        )
        return logits_student, {
            "loss_ce": loss_ce.unsqueeze(0),
            "loss_kd": loss_feat.unsqueeze(0),
            "loss_total": (loss_ce + loss_feat).unsqueeze(0),
        }
