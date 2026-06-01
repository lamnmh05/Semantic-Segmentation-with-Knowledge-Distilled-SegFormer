import torch
import torch.nn.functional as F

from .base import Distiller
from .mlp import MLPTransform


class Combine(Distiller):
    """Combined MLP feature distillation + BPKD for semantic segmentation."""

    @staticmethod
    def _to_scalar(value, name):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be a scalar value, got tensor with shape {tuple(value.shape)}")
            return float(value.item())

        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"{name} must be a scalar value, got {value!r}")
            value = value[0]

        return float(value)

    @staticmethod
    def _to_bool(value, name):
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in {"true", "1", "yes", "y", "on"}:
                return True
            if value_lower in {"false", "0", "no", "n", "off"}:
                return False

        raise ValueError(f"{name} must be a boolean value, got {value!r}")

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)

        distill = cfg["distill"]

        self.ce_loss_weight = self._to_scalar(
            distill.get("lambda_ce", distill.get("ce_weight", 1.0)),
            "lambda_ce",
        )
        self.mlp_loss_weight = self._to_scalar(
            distill.get("lambda_mlp", distill.get("feat_weight", 1.0)),
            "lambda_mlp",
        )
        self.bpkd_loss_weight = self._to_scalar(
            distill.get("lambda_bpkd", distill.get("kd_weight", 1.0)),
            "lambda_bpkd",
        )

        self.lambda_body = self._to_scalar(distill.get("lambda_body", 20.0), "lambda_body")
        self.lambda_edge = self._to_scalar(distill.get("lambda_edge", 50.0), "lambda_edge")
        self.alpha_edge = self._to_scalar(distill.get("alpha_edge", 2.0), "alpha_edge")
        self.temperature = self._to_scalar(distill.get("temperature", 4.0), "temperature")
        self.ignore_index = int(distill.get("ignore_index", 255))

        # ------------------------------------------------------------------
        # New options for combining MLP feature loss and BPKD.
        #
        # 1) balance_mode = "static":
        #       use fixed lambda_mlp and lambda_bpkd exactly like the old code.
        #
        # 2) balance_mode = "grad_norm":
        #       dynamically scale MLP and BPKD by inverse gradient norm:
        #           weight_i = lambda_i * mean_grad_norm / (grad_norm_i + eps)
        #       This is a light GradNorm-style normalization computed on:
        #           - projected student feature for MLP loss
        #           - student logits for BPKD loss
        #
        # Curriculum can be enabled together with either mode. The typical
        # schedule is: CE always on, MLP ramps early, BPKD starts later.
        # ------------------------------------------------------------------
        self.balance_mode = str(distill.get("balance_mode", "static")).lower()
        if self.balance_mode not in {"static", "grad_norm"}:
            raise ValueError(
                "distill.balance_mode must be either 'static' or 'grad_norm', "
                f"got {self.balance_mode!r}"
            )

        grad_norm_cfg = distill.get("grad_norm", {})
        self.grad_norm_eps = self._to_scalar(
            grad_norm_cfg.get("eps", distill.get("grad_norm_eps", 1.0e-8)),
            "grad_norm.eps",
        )
        self.grad_norm_min_scale = self._to_scalar(
            grad_norm_cfg.get("min_scale", distill.get("grad_norm_min_scale", 1.0e-6)),
            "grad_norm.min_scale",
        )
        self.grad_norm_max_scale = self._to_scalar(
            grad_norm_cfg.get("max_scale", distill.get("grad_norm_max_scale", 10.0)),
            "grad_norm.max_scale",
        )

        curriculum_cfg = distill.get("curriculum", {})
        self.curriculum_enabled = self._to_bool(
            curriculum_cfg.get("enabled", distill.get("curriculum_enabled", False)),
            "curriculum.enabled",
        )
        self.mlp_start_iter = int(
            self._to_scalar(curriculum_cfg.get("mlp_start_iter", 0), "curriculum.mlp_start_iter")
        )
        self.mlp_ramp_iters = int(
            self._to_scalar(curriculum_cfg.get("mlp_ramp_iters", 0), "curriculum.mlp_ramp_iters")
        )
        self.mlp_start_factor = self._to_scalar(
            curriculum_cfg.get("mlp_start_factor", 1.0),
            "curriculum.mlp_start_factor",
        )
        self.mlp_end_factor = self._to_scalar(
            curriculum_cfg.get("mlp_end_factor", 1.0),
            "curriculum.mlp_end_factor",
        )
        self.bpkd_start_iter = int(
            self._to_scalar(curriculum_cfg.get("bpkd_start_iter", 0), "curriculum.bpkd_start_iter")
        )
        self.bpkd_ramp_iters = int(
            self._to_scalar(curriculum_cfg.get("bpkd_ramp_iters", 0), "curriculum.bpkd_ramp_iters")
        )
        self.bpkd_start_factor = self._to_scalar(
            curriculum_cfg.get("bpkd_start_factor", 1.0),
            "curriculum.bpkd_start_factor",
        )
        self.bpkd_end_factor = self._to_scalar(
            curriculum_cfg.get("bpkd_end_factor", 1.0),
            "curriculum.bpkd_end_factor",
        )
        self._internal_iter = 0

        edge_width = distill.get("edge_width", None)
        if edge_width is None:
            self.boundary_radius = int(
                self._to_scalar(distill.get("boundary_radius", 3), "boundary_radius")
            )
            self.edge_width = 2 * self.boundary_radius + 1
        else:
            self.edge_width = int(self._to_scalar(edge_width, "edge_width"))
            if self.edge_width % 2 == 0:
                raise ValueError("edge_width must be odd, e.g. 3, 5, 7, 9.")
            self.boundary_radius = self.edge_width // 2

        self.feat_layer = distill.get("feat_layer", 2)

        student_channels = distill.get("student_channels", None)
        teacher_channels = distill.get("teacher_channels", None)

        if student_channels is None:
            raise ValueError("Missing cfg['distill']['student_channels'].")

        if teacher_channels is None:
            raise ValueError("Missing cfg['distill']['teacher_channels'].")

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

        device = next(self.student.parameters()).device
        self.mlp.to(device)

    def get_mlp_parameters(self):
        return list(self.mlp.parameters())

    def get_regressor_parameters(self):
        return self.get_mlp_parameters()

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + self.get_mlp_parameters()

    def get_extra_parameters(self):
        return sum(p.numel() for p in self.mlp.parameters())

    def _resize_logits(self, logits, size):
        if logits.shape[-2:] != size:
            logits = F.interpolate(
                logits,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def _pool_to_logits(self, x, logits_size):
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

    def _prepare_target(self, target):
        if target.dim() == 4:
            target = target.squeeze(1)

        if target.dim() != 3:
            raise ValueError(
                f"target must have shape [B, H, W] or [B, 1, H, W], got {target.shape}"
            )

        return target.long()

    def feature_loss_with_projection(self, s_feats, t_feats):
        idx = int(self.feat_layer)

        f_s = s_feats[idx]
        f_t = t_feats[idx]

        f_s_proj = self.mlp(f_s)

        if f_s_proj.shape[2:] != f_t.shape[2:]:
            f_s_proj = F.interpolate(
                f_s_proj,
                size=f_t.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        diff = (f_s_proj - f_t).pow(2)
        loss_feat = diff.sum() / diff.size(0)

        return loss_feat, f_s_proj

    def feature_loss(self, s_feats, t_feats):
        loss_feat, _ = self.feature_loss_with_projection(s_feats, t_feats)
        return loss_feat


    def get_edge_mask(self, target, num_classes, logits_size):
        target = self._prepare_target(target)

        B, H, W = target.shape

        valid_mask = target != self.ignore_index

        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        gt_onehot = F.one_hot(
            target_safe,
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).float()

        valid_mask_float = valid_mask.unsqueeze(1).float()
        gt_onehot = gt_onehot * valid_mask_float

        kernel_size = self.edge_width
        padding = self.boundary_radius

        dilation = F.max_pool2d(
            gt_onehot,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        erosion = 1.0 - F.max_pool2d(
            1.0 - gt_onehot,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        gt_edge = (dilation - erosion).clamp(min=0.0, max=1.0)
        gt_edge = gt_edge * valid_mask_float

        edge_mask = self._pool_to_logits(gt_edge, logits_size).clamp(min=0.0, max=1.0)
        valid_mask_logits = self._pool_to_logits(valid_mask_float, logits_size).clamp(
            min=0.0, max=1.0
        )

        edge_mask = edge_mask * valid_mask_logits

        return edge_mask, valid_mask_logits

    def edge_loss(self, logits_student, logits_teacher, edge_mask):
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

        z_s_edge = logits_student * edge_mask
        z_t_edge = logits_teacher.detach() * edge_mask

        log_prob_s = F.log_softmax(z_s_edge / T, dim=1)
        prob_t = F.softmax(z_t_edge / T, dim=1)

        kl_map_class = F.kl_div(
            log_prob_s,
            prob_t,
            reduction="none",
        ) * (T * T)

        phi_pixel = kl_map_class.sum(dim=1, keepdim=True)
        phi_repeat = phi_pixel.expand(-1, C, -1, -1)

        n_c = (edge_mask > 0).float().sum(dim=(2, 3)).clamp_min(1.0)
        loss_per_class = (phi_repeat * edge_mask).sum(dim=(2, 3)) / n_c
        loss_edge = loss_per_class.sum(dim=1).mean()

        return self.alpha_edge * loss_edge

    def body_loss(self, logits_student, logits_teacher, edge_mask, valid_mask_logits):
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

        body_mask = (1.0 - edge_mask) * valid_mask_logits

        z_s_body = logits_student * body_mask
        z_t_body = logits_teacher.detach() * body_mask

        z_s_body = z_s_body.flatten(2)
        z_t_body = z_t_body.flatten(2)

        log_prob_s = F.log_softmax(z_s_body / T, dim=2)
        prob_t = F.softmax(z_t_body / T, dim=2)

        kl_map = F.kl_div(
            log_prob_s,
            prob_t,
            reduction="none",
        )

        return (T * T) * kl_map.sum(dim=2).sum(dim=1).mean() / C

    def bpkd_loss(self, logits_student, logits_teacher, target):
        if logits_student.dim() != 4:
            raise ValueError(
                f"logits_student must have shape [B, C, H, W], got {logits_student.shape}"
            )

        if logits_teacher.dim() != 4:
            raise ValueError(
                f"logits_teacher must have shape [B, C, H, W], got {logits_teacher.shape}"
            )

        logits_size = logits_student.shape[-2:]
        logits_teacher = self._resize_logits(logits_teacher, logits_size)
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


    def _get_global_iter(self, kwargs):
        """Return current training iteration.

        The training loop can pass one of these names:
            iter, iteration, global_iter, global_step, step

        If none is passed, the distiller keeps an internal counter. This makes
        the curriculum usable even when the current trainer does not pass step.
        """
        for key in ("iter", "iteration", "global_iter", "global_step", "step"):
            if key in kwargs and kwargs[key] is not None:
                value = kwargs[key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                return int(value)

        self._internal_iter += 1
        return self._internal_iter

    @staticmethod
    def _linear_factor(step, start_iter, ramp_iters, start_factor, end_factor):
        if ramp_iters <= 0:
            return end_factor if step >= start_iter else start_factor

        if step < start_iter:
            return start_factor

        progress = (step - start_iter) / float(ramp_iters)
        progress = max(0.0, min(1.0, progress))

        return start_factor + progress * (end_factor - start_factor)

    def _curriculum_factors(self, step):
        if not self.curriculum_enabled:
            return 1.0, 1.0

        mlp_factor = self._linear_factor(
            step=step,
            start_iter=self.mlp_start_iter,
            ramp_iters=self.mlp_ramp_iters,
            start_factor=self.mlp_start_factor,
            end_factor=self.mlp_end_factor,
        )

        bpkd_factor = self._linear_factor(
            step=step,
            start_iter=self.bpkd_start_iter,
            ramp_iters=self.bpkd_ramp_iters,
            start_factor=self.bpkd_start_factor,
            end_factor=self.bpkd_end_factor,
        )

        return mlp_factor, bpkd_factor

    def _safe_tensor_grad_norm(self, loss, tensor):
        if (not torch.is_tensor(loss)) or (not loss.requires_grad):
            return tensor.new_tensor(0.0)

        grad = torch.autograd.grad(
            loss,
            tensor,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]

        if grad is None:
            return tensor.new_tensor(0.0)

        return grad.detach().float().norm(p=2)

    def _dynamic_loss_weights(
        self,
        loss_mlp_raw,
        f_s_proj,
        loss_bpkd_raw,
        logits_student,
        mlp_factor,
        bpkd_factor,
    ):
        """Return final weights for MLP and BPKD losses.

        In static mode:
            lambda_mlp_final  = lambda_mlp  * curriculum_mlp_factor
            lambda_bpkd_final = lambda_bpkd * curriculum_bpkd_factor

        In grad_norm mode:
            dynamic_scale_i = mean(active_grad_norms) / (grad_norm_i + eps)

        The dynamic scale is detached on purpose. It changes loss magnitude but
        does not create a second-order optimization objective.
        """
        base_mlp = self.mlp_loss_weight * mlp_factor
        base_bpkd = self.bpkd_loss_weight * bpkd_factor

        if self.balance_mode != "grad_norm":
            return (
                loss_mlp_raw.new_tensor(base_mlp),
                loss_bpkd_raw.new_tensor(base_bpkd),
                loss_mlp_raw.new_tensor(1.0),
                loss_bpkd_raw.new_tensor(1.0),
                loss_mlp_raw.new_tensor(0.0),
                loss_bpkd_raw.new_tensor(0.0),
            )

        g_mlp = self._safe_tensor_grad_norm(loss_mlp_raw, f_s_proj)
        g_bpkd = self._safe_tensor_grad_norm(loss_bpkd_raw, logits_student)

        active_grads = []
        if base_mlp > 0:
            active_grads.append(g_mlp)
        if base_bpkd > 0:
            active_grads.append(g_bpkd)

        if len(active_grads) == 0:
            mean_grad = loss_mlp_raw.new_tensor(1.0)
        else:
            mean_grad = torch.stack(active_grads).mean().detach().clamp_min(self.grad_norm_eps)

        mlp_scale = mean_grad / (g_mlp.detach() + self.grad_norm_eps)
        bpkd_scale = mean_grad / (g_bpkd.detach() + self.grad_norm_eps)

        mlp_scale = mlp_scale.clamp(
            min=self.grad_norm_min_scale,
            max=self.grad_norm_max_scale,
        )
        bpkd_scale = bpkd_scale.clamp(
            min=self.grad_norm_min_scale,
            max=self.grad_norm_max_scale,
        )

        weight_mlp = loss_mlp_raw.new_tensor(base_mlp) * mlp_scale
        weight_bpkd = loss_bpkd_raw.new_tensor(base_bpkd) * bpkd_scale

        return weight_mlp, weight_bpkd, mlp_scale, bpkd_scale, g_mlp, g_bpkd


    def forward_train(self, image, target, **kwargs):
        target = self._prepare_target(target)
        step = self._get_global_iter(kwargs)

        s_feats, _, logits_student = self.student.extract_feature(image)

        with torch.no_grad():
            t_feats, _, logits_teacher = self.teacher.extract_feature(image)

        logits_student_for_ce = self._resize_logits(
            logits_student,
            target.shape[-2:],
        )

        loss_ce_raw = F.cross_entropy(
            logits_student_for_ce,
            target,
            ignore_index=self.ignore_index,
        )
        loss_ce = self.ce_loss_weight * loss_ce_raw

        # MLP feature distillation.
        # f_s_proj is returned so grad_norm mode can measure ||dL_mlp / d f_s_proj||.
        loss_mlp_raw, f_s_proj = self.feature_loss_with_projection(s_feats, t_feats)

        # BPKD logit distillation.
        # loss_dict["loss_bpkd"] already includes lambda_body and lambda_edge,
        # but not lambda_bpkd. That outer lambda is handled below.
        loss_dict = self.bpkd_loss(
            logits_student=logits_student,
            logits_teacher=logits_teacher,
            target=target,
        )
        loss_bpkd_raw = loss_dict["loss_bpkd"]

        mlp_factor, bpkd_factor = self._curriculum_factors(step)

        (
            weight_mlp,
            weight_bpkd,
            scale_mlp,
            scale_bpkd,
            grad_norm_mlp,
            grad_norm_bpkd,
        ) = self._dynamic_loss_weights(
            loss_mlp_raw=loss_mlp_raw,
            f_s_proj=f_s_proj,
            loss_bpkd_raw=loss_bpkd_raw,
            logits_student=logits_student,
            mlp_factor=mlp_factor,
            bpkd_factor=bpkd_factor,
        )

        loss_mlp = weight_mlp * loss_mlp_raw
        loss_bpkd = weight_bpkd * loss_bpkd_raw

        loss_kd = loss_mlp + loss_bpkd
        loss_total = loss_ce + loss_kd

        return logits_student_for_ce, {
            "loss_ce": loss_ce,
            "loss_ce_raw": loss_ce_raw.detach(),
            "loss_mlp": loss_mlp,
            "loss_mlp_raw": loss_mlp_raw.detach(),
            "loss_body": loss_dict["loss_body"],
            "loss_edge": loss_dict["loss_edge"],
            "loss_body_w": loss_dict["loss_body_w"],
            "loss_edge_w": loss_dict["loss_edge_w"],
            "loss_bpkd": loss_bpkd,
            "loss_bpkd_raw": loss_bpkd_raw.detach(),
            "loss_kd": loss_kd,
            "loss_total": loss_total,

            # Useful logs for debugging the two new strategies.
            "distill_iter": torch.as_tensor(step, device=loss_total.device),
            "mlp_factor": torch.as_tensor(mlp_factor, device=loss_total.device),
            "bpkd_factor": torch.as_tensor(bpkd_factor, device=loss_total.device),
            "weight_mlp": weight_mlp.detach(),
            "weight_bpkd": weight_bpkd.detach(),
            "scale_mlp": scale_mlp.detach(),
            "scale_bpkd": scale_bpkd.detach(),
            "grad_norm_mlp": grad_norm_mlp.detach(),
            "grad_norm_bpkd": grad_norm_bpkd.detach(),
        }
