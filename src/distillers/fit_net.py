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
        
        self.ce_loss_weight = float(distill.get("ce_weight", 1.0))
        self.feat_loss_weight = float(distill.get("feat_weight", 10.0))
        self.hint_layer = int(distill.get("hint_layer", 2))

        img_size = cfg["dataset"]["img_size"]
        h, w = (img_size[0], img_size[1]) if isinstance(img_size, (list, tuple)) else (img_size, img_size)
        device = next(self.student.parameters()).device
        
        self.student.eval()
        self.teacher.eval()
        dummy_input = torch.randn(2, 3, h, w, device=device)

        with torch.no_grad():
            s_out = self.student.extract_feature(dummy_input)
            t_out = self.teacher.extract_feature(dummy_input)
            s_feats = self._parse_feats(s_out)
            t_feats = self._parse_feats(t_out)

        if self.hint_layer >= len(s_feats) or self.hint_layer >= len(t_feats):
            raise ValueError(f"hint_layer={self.hint_layer} out of range")

        s_channel = s_feats[self.hint_layer].shape[1]
        t_channel = t_feats[self.hint_layer].shape[1]
        self.conv_reg = ConvReg(s_channel, t_channel).to(device)

    def _parse_feats(self, output):
        """Robustly parse features from model output."""
        if isinstance(output, dict):
            feats = output.get('feats', output.get('features', []))
        elif isinstance(output, (list, tuple)):
            if len(output) > 0 and isinstance(output[0], (list, tuple)):
                feats = output[0]
            else:
                feats = list(output[:-1]) 
        else:
            raise ValueError("Unsupported output format from extract_feature")
            
        if isinstance(feats, dict): 
            feats = list(feats.values())
        if not isinstance(feats, (list, tuple)): 
            feats = [feats]
        return feats

    def get_regressor_parameters(self):
        return list(self.conv_reg.parameters())

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + self.get_regressor_parameters()

    def get_extra_parameters(self):
        return sum(p.numel() for p in self.conv_reg.parameters())

    def forward_train(self, image, target, stage="kd", **kwargs):
        s_out = self.student.extract_feature(image)
        s_feats = self._parse_feats(s_out)
        
        if isinstance(s_out, dict):
            logits_student = s_out.get('logits')
        elif isinstance(s_out, (list, tuple)):
            logits_student = s_out[-1]
        else:
            logits_student = s_out

        with torch.no_grad():
            t_out = self.teacher.extract_feature(image)
            t_feats = self._parse_feats(t_out)

        f_s = self.conv_reg(s_feats[self.hint_layer])
        f_t = t_feats[self.hint_layer]
        
        if f_s.shape[2:] != f_t.shape[2:]:
            f_s = F.interpolate(f_s, size=f_t.shape[2:], mode="bilinear", align_corners=False)

        loss_feat = self.feat_loss_weight * F.mse_loss(f_s, f_t)

        if stage == "hint":
            return None, {
                "loss_ce": loss_feat.new_zeros(()),
                "loss_kd": loss_feat,
                "loss_total": loss_feat,
            }

        if logits_student.shape[2:] != target.shape[1:]:
            logits_student = F.interpolate(logits_student, size=target.shape[1:], mode="bilinear", align_corners=False)
            
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target.long(), ignore_index=255)
        
        return logits_student, {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
            "loss_total": loss_ce + loss_feat,
        }