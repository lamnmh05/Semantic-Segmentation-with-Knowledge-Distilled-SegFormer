import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller


class MLPTransform(nn.Module):
    def __init__(self, s_channel, t_channel, hidden_channel=None):
        super().__init__()
        if hidden_channel is None:
            hidden_channel = t_channel

        self.net = nn.Sequential(
            nn.Conv2d(s_channel, hidden_channel, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, t_channel, kernel_size=1, bias=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class MLPFD(Distiller):
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        distill = cfg["distill"]
        
        self.ce_loss_weight = float(distill.get("ce_weight", 1.0))
        self.feat_loss_weight = float(distill.get("feat_weight", 1.0))
        self.ignore_index = int(distill.get("ignore_index", 255))
        self.feat_layer_idx = int(distill.get("feat_layer", 2))

        img_size = cfg["dataset"]["img_size"]
        h, w = (img_size[0], img_size[1]) if isinstance(img_size, (list, tuple)) else (img_size, img_size)
        device = next(self.student.parameters()).device
        
        self.student.eval()
        self.teacher.eval()
        dummy = torch.randn(1, 3, h, w, device=device)
        
        with torch.no_grad():
            s_out = self.student.extract_feature(dummy)
            t_out = self.teacher.extract_feature(dummy)
            
            s_feats = self._parse_feats(s_out)
            t_feats = self._parse_feats(t_out)

        if self.feat_layer_idx >= len(s_feats) or self.feat_layer_idx >= len(t_feats):
            raise ValueError(f"feat_layer_idx={self.feat_layer_idx} out of range.")

        s_ch = s_feats[self.feat_layer_idx].shape[1]
        t_ch = t_feats[self.feat_layer_idx].shape[1]
        
        hidden_ch = distill.get("mlp_hidden_channel", t_ch)
        self.mlp = MLPTransform(s_ch, t_ch, hidden_ch).to(device)

    def _parse_feats(self, output):
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

    def get_mlp_parameters(self):
        return list(self.mlp.parameters())

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + self.get_mlp_parameters()

    def get_extra_parameters(self):
        return sum(p.numel() for p in self.mlp.parameters())

    def feature_loss(self, s_feats, t_feats):
        idx = int(self.feat_layer_idx)
        
        f_s = s_feats[idx]
        f_t = t_feats[idx]

        f_s_proj = self.mlp(f_s)

        if f_s_proj.shape[2:] != f_t.shape[2:]:
            f_s_proj = F.interpolate(f_s_proj, size=f_t.shape[2:], mode="bilinear", align_corners=False)

        diff = (f_s_proj - f_t).pow(2)
        return diff.sum() / diff.size(0)

    def forward_train(self, image, target, **kwargs):
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

        loss_feat = self.feat_loss_weight * self.feature_loss(s_feats, t_feats)

        if logits_student.shape[2:] != target.shape[1:]:
            logits_student = F.interpolate(logits_student, size=target.shape[1:], mode="bilinear", align_corners=False)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target.long(), ignore_index=self.ignore_index)

        loss_total = loss_ce + loss_feat

        return logits_student, {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
            "loss_total": loss_total,
        }