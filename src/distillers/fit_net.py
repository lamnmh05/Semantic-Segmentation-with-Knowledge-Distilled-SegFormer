import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Distiller

class ConvReg(nn.Module):
    """Convolutional regressor for FitNet."""
    def __init__(self, s_channel, t_channel, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(s_channel, t_channel, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(t_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

def soft_kd_loss(logits_s, logits_t, temperature):
    """KL Divergence loss with temperature scaling."""
    if logits_s.shape[2:] != logits_t.shape[2:]:
        logits_t = F.interpolate(logits_t, size=logits_s.shape[2:], mode="bilinear", align_corners=False)
    
    B, C, H, W = logits_s.shape
    # Flatten to [B*H*W, C]
    logits_s_flat = logits_s.permute(0, 2, 3, 1).reshape(-1, C)
    logits_t_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, C)
    
    p_s = F.log_softmax(logits_s_flat / temperature, dim=-1)
    p_t = F.softmax(logits_t_flat / temperature, dim=-1)
    
    # batchmean reduction is standard for KL div in distillation
    loss = F.kl_div(p_s, p_t, reduction="batchmean") * (temperature ** 2)
    return loss

class FitNet(Distiller):
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        distill = cfg["distill"]
        
        # Hyperparams
        self.temperature = distill.get("temperature", 3.0)
        self.lambda_kd = distill.get("lambda_kd", 1.0)
        self.ce_weight = distill.get("ce_weight", 1.0)
        self.hint_layer_idx = distill.get("hint_layer", 2)
        self.hint_weight_stage2 = distill.get("hint_weight_stage2", 0.1)
        
        # Setup Connector
        img_size = cfg["dataset"]["img_size"]
        h, w = (img_size[0], img_size[1]) if isinstance(img_size, (list, tuple)) else (img_size, img_size)
        device = next(self.student.parameters()).device
        
        # Dummy forward to get shapes
        self.student.eval()
        self.teacher.eval()
        dummy = torch.randn(1, 3, h, w, device=device)
        
        with torch.no_grad():
            # Ensure models have extract_feature method returning (feats, ..., logits)
            s_out = self.student.extract_feature(dummy)
            t_out = self.teacher.extract_feature(dummy)
            
            # Robust extraction depending on return type (tuple vs dict)
            s_feats = s_out[1] if isinstance(s_out, tuple) and len(s_out) > 1 else s_out.get('feats', [])
            t_feats = t_out[1] if isinstance(t_out, tuple) and len(t_out) > 1 else t_out.get('feats', [])
            
            # Convert to list if it's a dict values
            if isinstance(s_feats, dict): s_feats = list(s_feats.values())
            if isinstance(t_feats, dict): t_feats = list(t_feats.values())

        if self.hint_layer_idx >= len(s_feats) or self.hint_layer_idx >= len(t_feats):
            raise ValueError(f"Hint layer index {self.hint_layer_idx} out of bounds.")

        s_ch = s_feats[self.hint_layer_idx].shape[1]
        t_ch = t_feats[self.hint_layer_idx].shape[1]
        
        self.conv_reg = ConvReg(s_ch, t_ch).to(device)

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())

    def forward_train(self, image, target, stage="kd", **kwargs):
        # Forward pass
        s_out = self.student.extract_feature(image)
        s_feats = s_out[1] if isinstance(s_out, tuple) and len(s_out) > 1 else s_out.get('feats', [])
        logits_student = s_out[0] if isinstance(s_out, tuple) else s_out.get('logits')
        
        with torch.no_grad():
            t_out = self.teacher.extract_feature(image)
            t_feats = t_out[1] if isinstance(t_out, tuple) and len(t_out) > 1 else t_out.get('feats', [])
            logits_teacher = t_out[0] if isinstance(t_out, tuple) else t_out.get('logits')

        if isinstance(s_feats, dict): s_feats = list(s_feats.values())
        if isinstance(t_feats, dict): t_feats = list(t_feats.values())

        # Hint Loss Calculation
        f_s = s_feats[self.hint_layer_idx]
        f_t = t_feats[self.hint_layer_idx]
        
        f_s_reg = self.conv_reg(f_s)
        if f_s_reg.shape[2:] != f_t.shape[2:]:
            f_s_reg = F.interpolate(f_s_reg, size=f_t.shape[2:], mode="bilinear", align_corners=False)
        
        loss_hint = F.mse_loss(f_s_reg, f_t)

        # Stage 1: Hint Pre-training
        if stage == "hint":
            return None, {
                "loss_total": loss_hint,
                "loss_hint": loss_hint,
                "loss_ce": torch.tensor(0.0, device=image.device),
                "loss_kd": torch.tensor(0.0, device=image.device),
            }

        # Stage 2: Knowledge Distillation
        logits_s_up = F.interpolate(logits_student, size=target.shape[1:], mode="bilinear", align_corners=False)
        loss_ce = self.ce_weight * F.cross_entropy(logits_s_up, target, ignore_index=255)
        loss_kd = soft_kd_loss(logits_student, logits_teacher, self.temperature)
        
        # Total Loss
        loss_total = loss_ce + self.lambda_kd * loss_kd + self.hint_weight_stage2 * loss_hint

        return logits_s_up, {
            "loss_total": loss_total,
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_hint": loss_hint,
        }