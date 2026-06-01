import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedSegformer(nn.Module):
    def __init__(self, student, cfg):
        super().__init__()
        self.student = student
        self.ce_loss_weight = float((cfg.get("distill") or {}).get("ce_weight", 1.0))
        self.ignore_index = int(
            (cfg.get("distill") or {}).get("ignore_index", 255)
        )

    def get_learnable_parameters(self):
        return [param for param in self.student.parameters() if param.requires_grad]

    def _prepare_target(self, target):
        if target.dim() == 4:
            target = target.squeeze(1)
        if target.dim() != 3:
            raise ValueError(
                f"target must have shape [B, H, W] or [B, 1, H, W], got {target.shape}"
            )
        return target.long()

    def forward_train(self, image, target, **kwargs):
        target = self._prepare_target(target)

        _, _, logits_student = self.student.extract_feature(image)
        if logits_student.shape[-2:] != target.shape[-2:]:
            logits_student = F.interpolate(
                logits_student,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        loss_ce = self.ce_loss_weight * F.cross_entropy(
            logits_student,
            target,
            ignore_index=self.ignore_index,
        )

        return logits_student, {
            "loss_ce": loss_ce,
            "loss_kd": loss_ce.new_zeros(()),
            "loss_total": loss_ce,
        }

    def forward_test(self, image):
        return self.student(image)

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])
