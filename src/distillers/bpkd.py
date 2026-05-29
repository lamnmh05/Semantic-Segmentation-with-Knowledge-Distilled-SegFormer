import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller


class BPKD(Distiller):
    """
    Boundary Privileged Knowledge Distillation for Semantic Segmentation.
    Total loss:
        loss_total = ce_weight * CE
                   + kd_weight * Boundary_KL(student_logits, teacher_logits)
    """

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)

        distill = cfg["distill"]

        self.ce_loss_weight = float(distill.get("ce_weight", 1.0))
        self.kd_loss_weight = float(distill.get("kd_weight", 1.0))

        self.temperature = float(distill.get("temperature", 4.0))
        self.ignore_index = int(distill.get("ignore_index", 255))

        # boundary_radius = 1 nghĩa là làm dày boundary bằng kernel 3x3.
        # boundary_radius = 2 nghĩa là kernel 5x5.
        self.boundary_radius = distill.get("boundary_radius", 1)

        # Nếu True: chỉ tính KD ở boundary.
        # Nếu False: tính KD toàn ảnh nhưng boundary được tăng trọng số.
        self.boundary_only = distill.get("boundary_only", True)

        # Chỉ dùng khi boundary_only = False
        self.boundary_weight = distill.get("boundary_weight", 5.0)

    def get_extra_parameters(self):
        return 0

    def _resize_logits(self, logits, size):
        """
        Resize logits về cùng size với target.

        logits: [B, C, H, W]
        size:   [H_target, W_target]
        """
        if logits.shape[2:] != size:
            logits = F.interpolate(
                logits,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def get_boundary_mask(self, target):
        """
        target:
            [B, H, W]

        return:
            boundary_mask: [B, H, W], dtype=torch.bool

        Pixel được xem là boundary nếu nó có label khác với pixel lân cận.
        """

        if target.dim() == 4:
            target = target.squeeze(1)

        target = target.long()
        valid = target != self.ignore_index

        boundary = torch.zeros_like(valid, dtype=torch.bool)

        # So sánh theo chiều dọc
        diff_h = (target[:, 1:, :] != target[:, :-1, :])
        valid_h = valid[:, 1:, :] & valid[:, :-1, :]
        diff_h = diff_h & valid_h

        boundary[:, 1:, :] |= diff_h
        boundary[:, :-1, :] |= diff_h

        # So sánh theo chiều ngang
        diff_w = (target[:, :, 1:] != target[:, :, :-1])
        valid_w = valid[:, :, 1:] & valid[:, :, :-1]
        diff_w = diff_w & valid_w

        boundary[:, :, 1:] |= diff_w
        boundary[:, :, :-1] |= diff_w

        # Làm dày boundary để vùng KD không quá mỏng
        if self.boundary_radius > 0:
            kernel_size = 2 * self.boundary_radius + 1

            boundary = boundary.float().unsqueeze(1)

            boundary = F.max_pool2d(
                boundary,
                kernel_size=kernel_size,
                stride=1,
                padding=self.boundary_radius,
            )

            boundary = boundary.squeeze(1).bool()

        boundary = boundary & valid

        return boundary

    def bpkd_loss(self, logits_student, logits_teacher, target):
        """
        Boundary Privileged KD loss.

        logits_student:
            [B, C, H, W]

        logits_teacher:
            [B, C, H, W]

        target:
            [B, H, W]
        """

        target_size = target.shape[-2:]

        logits_student = self._resize_logits(logits_student, target_size)
        logits_teacher = self._resize_logits(logits_teacher, target_size)

        T = self.temperature

        log_prob_student = F.log_softmax(logits_student / T, dim=1)
        prob_teacher = F.softmax(logits_teacher / T, dim=1)

        # KL theo từng pixel
        # output ban đầu: [B, C, H, W]
        # sum theo class -> [B, H, W]
        pixel_kd = F.kl_div(
            log_prob_student,
            prob_teacher,
            reduction="none",
        ).sum(dim=1) * (T * T)

        valid_mask = target != self.ignore_index
        boundary_mask = self.get_boundary_mask(target)

        if self.boundary_only:
            # Chỉ học KD tại boundary
            mask = boundary_mask.float()
            denom = mask.sum().clamp_min(1.0)
            loss = (pixel_kd * mask).sum() / denom
        else:
            # Học KD toàn ảnh, nhưng boundary có trọng số cao hơn
            weight = valid_mask.float()
            weight = weight + self.boundary_weight * boundary_mask.float()

            denom = weight.sum().clamp_min(1.0)
            loss = (pixel_kd * weight).sum() / denom

        return loss

    def forward_train(self, image, target, **kwargs):
        """
        Training forward.

        image:
            [B, 3, H, W]

        target:
            [B, H, W]
        """

        _, _, logits_student = self.student.extract_feature(image)

        with torch.no_grad():
            _, _, logits_teacher = self.teacher.extract_feature(image)

        target_size = target.shape[-2:]

        logits_student = self._resize_logits(logits_student, target_size)
        logits_teacher = self._resize_logits(logits_teacher, target_size)

        loss_ce = self.ce_loss_weight * F.cross_entropy(
            logits_student,
            target.long(),
            ignore_index=self.ignore_index,
        )

        loss_kd = self.kd_loss_weight * self.bpkd_loss(
            logits_student=logits_student,
            logits_teacher=logits_teacher,
            target=target,
        )

        loss_total = loss_ce + loss_kd

        return logits_student, {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_total": loss_total,
        }