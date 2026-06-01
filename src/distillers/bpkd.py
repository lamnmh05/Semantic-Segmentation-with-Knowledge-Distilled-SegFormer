import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller


class BPKD(Distiller):
    """
    Boundary Privileged Knowledge Distillation for Semantic Segmentation.

    Paper pipeline:

        1. Edge mask:
            GT_edge = dilation(GT) - erosion(GT)
            M_E = AvgPool2D(GT_edge)

        2. Decouple logits:
            Z_E = Z * M_E
            Z_B = Z * (1 - M_E)

        3. Edge loss:
            PRM:
                Z_E^S = Z^S * M_E
                Z_E^T = Z^T * M_E

            POM:
                L_E = sum_c alpha / n_c * sum_i phi_i * M_E,c,i

        4. Body loss:
            Channel-wise distillation on spatial dimension.

        5. Final:
            L_BPKD = lambda_body * L_B + lambda_edge * L_E
            L_total = CE + L_BPKD
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

        self.edge_width = distill.get("edge_width", 7)

        if self.edge_width % 2 == 0:
            raise ValueError("edge_width must be odd, e.g. 3, 5, 7, 9.")

        self.boundary_radius = self.edge_width // 2

    def get_extra_parameters(self):
        return 0

    def _prepare_target(self, target):
        """
        Convert target to [B, H, W].
        """
        if target.dim() == 4:
            target = target.squeeze(1)

        if target.dim() != 3:
            raise ValueError(
                f"target must be [B,H,W] or [B,1,H,W], got {target.shape}"
            )

        return target.long()

    def _resize_logits(self, logits, size):
        """
        Resize logits to given spatial size.
        """
        if logits.shape[-2:] != size:
            logits = F.interpolate(
                logits,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def _pool_to_logits(self, x, logits_size):
        """
        Downsample mask from GT size to logits size using AvgPool2D,
        following the paper.
        """
        H, W = x.shape[-2:]
        H_logit, W_logit = logits_size

        if H == H_logit and W == W_logit:
            return x

        if H % H_logit == 0 and W % W_logit == 0:
            stride_h = H // H_logit
            stride_w = W // W_logit

            return F.avg_pool2d(
                x,
                kernel_size=(stride_h, stride_w),
                stride=(stride_h, stride_w),
            )

        return F.adaptive_avg_pool2d(x, output_size=logits_size)

    def get_edge_mask(self, target, num_classes, logits_size):
        """
        Generate soft edge mask M_E.

        Args:
            target:
                [B, H, W]
            num_classes:
                C
            logits_size:
                [H', W']

        Returns:
            edge_mask:
                M_E, [B, C, H', W']
            valid_mask_logits:
                [B, 1, H', W']
        """

        target = self._prepare_target(target)

        B, H, W = target.shape

        valid_mask = target != self.ignore_index  # [B, H, W]

        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        # One-hot GT: [B,H,W] -> [B,C,H,W]
        gt_onehot = F.one_hot(
            target_safe,
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).float()

        valid_mask_float = valid_mask.unsqueeze(1).float()

        # Remove ignore region from GT one-hot
        gt_onehot = gt_onehot * valid_mask_float

        # Adjustable Trimap
        kernel_size = self.edge_width
        padding = self.boundary_radius

        # Dilation
        dilation = F.max_pool2d(
            gt_onehot,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        # Erosion for binary mask:
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

        # Remove ignore region
        gt_edge = gt_edge * valid_mask_float

        # Paper: M_E = AvgPool2D(GT_edge)
        edge_mask = self._pool_to_logits(gt_edge, logits_size)
        edge_mask = edge_mask.clamp(min=0.0, max=1.0)

        # Downsample valid mask too, so KD ignores ignore_index regions
        valid_mask_logits = self._pool_to_logits(valid_mask_float, logits_size)
        valid_mask_logits = valid_mask_logits.clamp(min=0.0, max=1.0)

        edge_mask = edge_mask * valid_mask_logits

        return edge_mask, valid_mask_logits

    def edge_loss(self, logits_student, logits_teacher, edge_mask):
        """
        Edge Knowledge Representation.

        Paper Figure 2:
            Pre-Mask Filtering
            -> Softmax(C)
            -> KL divergence
            -> Sum(C)
            -> Repeat(C)
            -> Post-Mask Filtering
            -> Apply alpha
            -> Sum / mask pixels
        """

        T = self.temperature

        B, C, H, W = logits_student.shape

        edge_mask = edge_mask.to(
            device=logits_student.device,
            dtype=logits_student.dtype,
        )

        if edge_mask.shape[-2:] != (H, W):
            edge_mask = F.interpolate(
                edge_mask,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        # PRM: Pre-Mask Filtering
        z_s_edge = logits_student * edge_mask
        z_t_edge = logits_teacher.detach() * edge_mask

        log_prob_s = F.log_softmax(z_s_edge / T, dim=1)
        prob_t = F.softmax(z_t_edge / T, dim=1)

        # KL per class: [B, C, H, W]
        kl_map_class = F.kl_div(
            log_prob_s,
            prob_t,
            reduction="none",
        ) * (T * T)

        # Paper Figure 2: Sum(C)
        # [B, C, H, W] -> [B, 1, H, W]
        phi_pixel = kl_map_class.sum(dim=1, keepdim=True)

        # Paper Figure 2: Repeat(nC)
        # [B, 1, H, W] -> [B, C, H, W]
        phi_repeat = phi_pixel.expand(-1, C, -1, -1)

        # POM: Post-Mask Filtering
        # n_c: number of non-zero pixels in M_E,c
        n_c = (edge_mask > 0).float().sum(dim=(2, 3)).clamp_min(1.0)

        # L_E,c = alpha / n_c * sum_i phi_i * M_E,c,i
        loss_per_class = (phi_repeat * edge_mask).sum(dim=(2, 3)) / n_c

        # Sum over classes, mean over batch
        loss_edge = loss_per_class.sum(dim=1).mean()

        # Edge loss inner weight alpha
        loss_edge = self.alpha_edge * loss_edge

        return loss_edge

    def body_loss(self, logits_student, logits_teacher, edge_mask, valid_mask_logits):
        """
        Body Knowledge Representation.

        Paper:
            Z_B = Z * (1 - M_E)

        Body loss:
            Channel-wise distillation.
            Softmax is applied on spatial dimension H'W',
            not on class dimension C.
        """

        T = self.temperature

        B, C, H, W = logits_student.shape

        edge_mask = edge_mask.to(
            device=logits_student.device,
            dtype=logits_student.dtype,
        )

        valid_mask_logits = valid_mask_logits.to(
            device=logits_student.device,
            dtype=logits_student.dtype,
        )

        if edge_mask.shape[-2:] != (H, W):
            edge_mask = F.interpolate(
                edge_mask,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        if valid_mask_logits.shape[-2:] != (H, W):
            valid_mask_logits = F.interpolate(
                valid_mask_logits,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        # Z_B = Z * (1 - M_E)
        body_mask = (1.0 - edge_mask) * valid_mask_logits

        z_s_body = logits_student * body_mask
        z_t_body = logits_teacher.detach() * body_mask

        # [B, C, H, W] -> [B, C, H*W]
        z_s_body = z_s_body.flatten(2)
        z_t_body = z_t_body.flatten(2)

        # Channel-wise distillation:
        # softmax over spatial dimension
        log_prob_s = F.log_softmax(z_s_body / T, dim=2)
        prob_t = F.softmax(z_t_body / T, dim=2)

        # KL per spatial position
        kl_map = F.kl_div(
            log_prob_s,
            prob_t,
            reduction="none",
        )

        # Paper: T^2 / C * sum_c sum_i KL
        loss_body = (T * T) * kl_map.sum(dim=2).sum(dim=1).mean() / C

        return loss_body

    def bpkd_loss(self, logits_student, logits_teacher, target):
        """
        Full BPKD loss:

            L_BPKD = lambda_body * L_B + lambda_edge * L_E
        """

        if logits_student.dim() != 4:
            raise ValueError(
                f"logits_student must be [B,C,H,W], got {logits_student.shape}"
            )

        if logits_teacher.dim() != 4:
            raise ValueError(
                f"logits_teacher must be [B,C,H,W], got {logits_teacher.shape}"
            )

        logits_size = logits_student.shape[-2:]

        logits_teacher = self._resize_logits(
            logits_teacher,
            logits_size,
        )

        num_classes = logits_student.shape[1]

        edge_mask, valid_mask_logits = self.get_edge_mask(
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
            valid_mask_logits=valid_mask_logits,
        )

        loss_body_w = self.lambda_body * loss_body
        loss_edge_w = self.lambda_edge * loss_edge

        loss_bpkd = loss_body_w + loss_edge_w

        return {
            "loss_bpkd": loss_bpkd,
            "loss_body": loss_body,
            "loss_edge": loss_edge,
            "loss_body_w": loss_body_w,
            "loss_edge_w": loss_edge_w,
        }

    def forward_train(self, image, target, **kwargs):
        """
        image:
            [B, 3, H, W]

        target:
            [B, H, W]
        """

        _, _, logits_student = self.student.extract_feature(image)

        with torch.no_grad():
            _, _, logits_teacher = self.teacher.extract_feature(image)

        target = self._prepare_target(target)

        # CE dùng logits resize về GT size
        logits_student_for_ce = self._resize_logits(
            logits_student,
            target.shape[-2:],
        )

        loss_ce = self.ce_loss_weight * F.cross_entropy(
            logits_student_for_ce,
            target,
            ignore_index=self.ignore_index,
        )

        # KD dùng logits gốc, mask được AvgPool về size logits
        loss_dict = self.bpkd_loss(
            logits_student=logits_student,
            logits_teacher=logits_teacher,
            target=target,
        )

        loss_kd = self.kd_loss_weight * loss_dict["loss_bpkd"]

        loss_total = loss_ce + loss_kd

        return logits_student_for_ce, {
            "loss_ce": loss_ce,
            "loss_body": loss_dict["loss_body"],
            "loss_edge": loss_dict["loss_edge"],
            "loss_body_w": loss_dict["loss_body_w"],
            "loss_edge_w": loss_dict["loss_edge_w"],
            "loss_kd": loss_kd,
            "loss_total": loss_total,
        }