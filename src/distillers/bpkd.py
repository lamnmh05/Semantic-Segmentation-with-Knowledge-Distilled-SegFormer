import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller


class BPKD(Distiller):
    """
    Boundary Privileged Knowledge Distillation for Semantic Segmentation.
        1. Edge mask:
            GT_edge = dilation(GT) - erosion(GT)
            M_E = AvgPool2D(GT_edge)

        2. Logit decoupling:
            Z_E = Z * M_E
            Z_B = Z * (1 - M_E)

        3. Distillation loss:
            loss_bpkd = lambda_body * loss_body
                      + lambda_edge * loss_edge

        4. Total loss:
            loss_total = ce_weight * CE
                       + kd_weight * loss_bpkd
    """

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)

        distill = cfg["distill"]

        self.ce_loss_weight = distill.get("ce_weight", 1.0)
        self.kd_loss_weight = distill.get("kd_weight", 1.0)

        self.lambda_body = distill.get("lambda_body", 20.0)
        self.lambda_edge = distill.get("lambda_edge", 50.0)
        self.alpha_edge = distill.get("alpha_edge", 2.0)


        self.temperature = distill.get("temperature", 4.0)
        self.ignore_index = distill.get("ignore_index", 255)

        # "adjustable Trimap"
        # boundary_radius = 1 -> kernel 3x3
        # boundary_radius = 2 -> kernel 5x5
        # boundary_radius = 3 -> kernel 7x7
        self.boundary_radius = distill.get("boundary_radius", 3)


    def get_extra_parameters(self):
        return 0

    def _resize_logits(self, logits, size):
        """
        [B, C, H, W] -> [B, C, H_new, W_new]
        """
        if logits.shape[-2:] != size:
            logits = F.interpolate(
                logits,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def _prepare_target(self, target):
        """
        Normalize target shape to [B, H, W].
        """
        if target.dim() == 4:
            target = target.squeeze(1)

        if target.dim() != 3:
            raise ValueError(
                f"target must have shape [B, H, W] or [B, 1, H, W], got {target.shape}"
            )

        return target.long()

    def get_edge_mask(self, target, num_classes, logits_size):
        """
        Get soft edge mask M_E

        Args:
            target:
                GT segmentation mask, shape [B, H, W].
            num_classes:
                Number of class C.
            logits_size:
                Spatial size of logits (H', W').

        Returns:
            edge_mask:
                M_E, shape [B, C, H', W'].
                between [0, 1].

        Pipeline:
            target [B,H,W]
                -> one-hot [B,C,H,W]
                -> dilation - erosion
                -> GT_edge [B,C,H,W]
                -> AvgPool2D
                -> M_E [B,C,H',W']
        """

        target = self._prepare_target(target)

        B, H, W = target.shape
        H_logit, W_logit = logits_size

        valid_mask = target != self.ignore_index  # [B, H, W]

        # Đổi ignore_index thành 0 tạm thời để one-hot không lỗi.
        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        # One-hot:
        # [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
        gt_onehot = F.one_hot(
            target_safe,
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).float()

        # Xóa vùng ignore khỏi one-hot.
        gt_onehot = gt_onehot * valid_mask.unsqueeze(1).float()

        # Kernel cho trimap boundary.
        kernel_size = 2 * self.boundary_radius + 1
        padding = self.boundary_radius

        # dilation(GT)
        dilation = F.max_pool2d(
            gt_onehot,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        # erosion(GT)
        # Với binary mask:
        # erosion(x) = 1 - dilation(1 - x)
        erosion = 1.0 - F.max_pool2d(
            1.0 - gt_onehot,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        # GT_edge = dilation(GT) - erosion(GT)
        gt_edge = dilation - erosion
        gt_edge = gt_edge.clamp(min=0.0, max=1.0)

        # Không tính edge ở vùng ignore.
        gt_edge = gt_edge * valid_mask.unsqueeze(1).float()

        # AvgPool2D để đưa GT_edge về cùng size với logits prediction.
        #
        # Paper:
        #   GT_edge: [B, C, H, W]
        #   M_E:     [B, C, H', W']
        #
        # Nếu H/H' và W/W' chia hết, dùng avg_pool2d đúng output stride.
        # Nếu không chia hết, dùng adaptive_avg_pool2d để đảm bảo shape.
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

        edge_mask = edge_mask.clamp(min=0.0, max=1.0)

        return edge_mask

    def edge_loss(self, logits_student, logits_teacher, edge_mask):
        """
        Edge Knowledge Representation.
        Args:
            logits_student:
                [B, C, H', W']
            logits_teacher:
                [B, C, H', W']
            edge_mask:
                M_E, [B, C, H', W']

        Returns:
            loss_edge:
                scalar tensor.

        PRM:
            Z_E^S = Z^S * M_E
            Z_E^T = Z^T * M_E

            phi_i = KL(
                softmax(Z_E^T_i / T),
                softmax(Z_E^S_i / T)
            )

        POM:
            L_E = sum_c alpha_c / n_c * sum_i phi_i * M_E,c,i
        """

        T = self.temperature

        B, C, H, W = logits_student.shape

        # PRM: mask logits trước khi tính KL.
        s_edge = logits_student * edge_mask
        t_edge = logits_teacher * edge_mask

        log_prob_s = F.log_softmax(s_edge / T, dim=1)
        prob_t = F.softmax(t_edge / T, dim=1)

        # KL theo từng pixel:
        # kl_map_class: [B, C, H, W]
        # kl_map_pixel: [B, 1, H, W]
        kl_map_class = F.kl_div(
            log_prob_s,
            prob_t,
            reduction="none",
        ) * (T * T)

        kl_map_pixel = kl_map_class.sum(dim=1, keepdim=True)

        # POM:
        # Lặp lại KL pixel cho từng class rồi nhân với mask của class đó.
        # edge_mask: [B, C, H, W]
        weighted_kl = kl_map_pixel * edge_mask

        # n_c: số pixel edge của từng class.
        # [B, C]
        n_c = edge_mask.sum(dim=(2, 3)).clamp_min(1.0)

        # [B, C]
        loss_per_class = weighted_kl.sum(dim=(2, 3)) / n_c

        loss_edge = self.alpha_edge * loss_per_class.sum(dim=1).mean()

        return loss_edge

    def body_loss(self, logits_student, logits_teacher, edge_mask):
        """
        Body Knowledge Representation.

        Paper:
            Z_B = Z * (1 - M_E)

        Body loss dùng channel-wise distillation:
            Với mỗi channel/class c, softmax trên spatial dimension H'W'.

        Args:
            logits_student:
                [B, C, H', W']
            logits_teacher:
                [B, C, H', W']
            edge_mask:
                [B, C, H', W']

        Returns:
            loss_body:
                scalar tensor.
        """

        T = self.temperature

        B, C, H, W = logits_student.shape

        body_mask = 1.0 - edge_mask

        s_body = logits_student * body_mask
        t_body = logits_teacher * body_mask

        # Flatten spatial dimension:
        # [B, C, H, W] -> [B, C, H*W]
        s_body = s_body.view(B, C, -1)
        t_body = t_body.view(B, C, -1)

        # Channel-wise distillation:
        # softmax trên spatial dimension, không phải class dimension.
        log_prob_s = F.log_softmax(s_body / T, dim=2)
        prob_t = F.softmax(t_body / T, dim=2)

        # KL theo spatial distribution của từng channel:
        # [B, C, H*W] -> sum spatial -> [B, C]
        kl_per_channel = F.kl_div(
            log_prob_s,
            prob_t,
            reduction="none",
        ).sum(dim=2)

        # Paper có hệ số T^2 / C.
        loss_body = (T * T) * kl_per_channel.sum(dim=1).mean() / C

        return loss_body

    def bpkd_loss(self, logits_student, logits_teacher, target):
        """
        Full BPKD loss:

            L_BPKD = lambda_body * L_B
                   + lambda_edge * L_E

        Lưu ý:
            - Không resize logits_student lên target size cho KD.
            - Thay vào đó, tạo edge_mask từ GT rồi AvgPool xuống size logits.
        """

        if logits_student.dim() != 4:
            raise ValueError(
                f"logits_student must have shape [B, C, H, W], got {logits_student.shape}"
            )

        if logits_teacher.dim() != 4:
            raise ValueError(
                f"logits_teacher must have shape [B, C, H, W], got {logits_teacher.shape}"
            )

        # Đưa teacher logits về cùng size với student logits nếu khác size.
        logits_size = logits_student.shape[-2:]
        logits_teacher = self._resize_logits(logits_teacher, logits_size)

        num_classes = logits_student.shape[1]

        # M_E: [B, C, H', W']
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

        # CE dùng logits đã resize về size target.
        logits_student_for_ce = self._resize_logits(
            logits_student,
            target.shape[-2:],
        )

        loss_ce = self.ce_loss_weight * F.cross_entropy(
            logits_student_for_ce,
            self._prepare_target(target),
            ignore_index=self.ignore_index,
        )

        # KD dùng logits gốc, mask sẽ được downsample về size logits.
        loss_bpkd_raw, loss_body_raw, loss_edge_raw = self.bpkd_loss(
            logits_student=logits_student,
            logits_teacher=logits_teacher,
            target=target,
        )

        loss_kd = self.kd_loss_weight * loss_bpkd_raw

        loss_total = loss_ce + loss_kd

        return logits_student_for_ce, {
            "loss_ce": loss_ce,
            "loss_body": loss_body_raw,
            "loss_edge": loss_edge_raw,
            "loss_kd": loss_kd,
            "loss_total": loss_total,
        }