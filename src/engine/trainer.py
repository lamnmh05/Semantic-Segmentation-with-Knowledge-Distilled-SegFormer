import os
import logging

import torch

from src.engine.lr_scheduler import get_lr, set_optimizer_lr
from src.eval import run_student_evaluation


def setup_logger(log_dir, name):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class Trainer:
    def __init__(self, distiller, train_loader, val_loader, optimizer, device, cfg):
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg

        train_cfg = cfg["train"]
        self.max_iters = train_cfg.get("max_iters")
        if self.max_iters is None:
            epochs = train_cfg.get("epochs", 50)
            self.max_iters = epochs * len(train_loader)
        self.eval_interval = train_cfg.get("eval_interval", 4000)
        self.log_interval = train_cfg.get("log_interval", 50)
        self.warmup_iters = train_cfg.get("warmup_iters", 1500)
        self.warmup_ratio = train_cfg.get("warmup_ratio", 1e-6)
        self.poly_power = train_cfg.get("poly_power", 1.0)
        self.base_lr = train_cfg["lr"]
        self.grad_clip = train_cfg.get("grad_clip")

        distill_cfg = cfg.get("distill", {})
        self.method = distill_cfg.get("method")
        self.connector_warmup_iters = 0
        if self.method == "AttnFD":
            warmup_epochs = train_cfg.get("connector_warmup_epochs", 1)
            self.connector_warmup_iters = warmup_epochs * len(train_loader)
        self.hint_pretrain_iters = 0
        if self.method == "FitNet":
            self.hint_pretrain_iters = distill_cfg.get("hint_pretrain_iters", 0)

        self.output_dir = cfg["experiment"]["output_dir"]
        self.logger = setup_logger(cfg["experiment"]["log_dir"], cfg["experiment"]["name"])
        os.makedirs(self.output_dir, exist_ok=True)

    def _is_connector_warmup(self, global_iter):
        return (
            self.method == "AttnFD"
            and self.connector_warmup_iters > 0
            and global_iter < self.connector_warmup_iters
        )

    def _is_hint_pretrain(self, global_iter):
        return self.hint_pretrain_iters > 0 and global_iter < self.hint_pretrain_iters

    def _set_trainable_for_phase(self, global_iter):
        for p in self.distiller.parameters():
            p.requires_grad = False

        if self._is_connector_warmup(global_iter):
            for p in self.distiller.get_connector_parameters():
                p.requires_grad = True
            return
        if self._is_hint_pretrain(global_iter):
            for p in self.distiller.get_regressor_parameters():
                p.requires_grad = True
            return

        for p in self.distiller.student.parameters():
            p.requires_grad = True
        if self.method == "AttnFD":
            for p in self.distiller.get_connector_parameters():
                p.requires_grad = True
        elif self.method == "FitNet":
            for p in self.distiller.get_regressor_parameters():
                p.requires_grad = True

    def train(self):
        self.logger.info(
            f"Starting training: max_iters={self.max_iters}, "
            f"method={self.method}, lr={self.base_lr}"
        )
        if self.connector_warmup_iters:
            self.logger.info(f"AttnFD connector warmup for {self.connector_warmup_iters} iters")
        if self.hint_pretrain_iters:
            self.logger.info(f"FitNet hint pretrain for {self.hint_pretrain_iters} iters")

        best_miou = 0.0
        global_iter = 0
        epoch = 0
        train_iter = iter(self.train_loader)

        while global_iter < self.max_iters:
            try:
                images, targets = next(train_iter)
            except StopIteration:
                epoch += 1
                train_iter = iter(self.train_loader)
                images, targets = next(train_iter)

            images = images.to(self.device)
            targets = targets.to(self.device)

            self._set_trainable_for_phase(global_iter)
            self.distiller.train()

            lr = get_lr(
                global_iter, self.base_lr, self.max_iters,
                warmup_iters=self.warmup_iters,
                warmup_ratio=self.warmup_ratio,
                power=self.poly_power,
            )
            set_optimizer_lr(self.optimizer, lr)

            self.optimizer.zero_grad()
            hint_only = self._is_hint_pretrain(global_iter)
            logits, losses = self.distiller.forward_train(
                images, targets, hint_only=hint_only
            )
            loss = losses["loss_total"]
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.distiller.parameters() if p.requires_grad],
                    self.grad_clip,
                )
            self.optimizer.step()

            if (global_iter + 1) % self.log_interval == 0:
                phase = "warmup" if self._is_connector_warmup(global_iter) else (
                    "hint" if hint_only else "train"
                )
                self.logger.info(
                    f"Iter [{global_iter + 1}/{self.max_iters}] epoch={epoch} phase={phase} "
                    f"lr={lr:.2e} loss={loss.item():.4f} "
                    f"ce={losses['loss_ce'].item():.4f} kd={losses['loss_kd'].item():.4f}"
                )

            should_eval = (
                self.val_loader
                and (global_iter + 1) % self.eval_interval == 0
                and not self._is_connector_warmup(global_iter)
                and not self._is_hint_pretrain(global_iter)
            )
            if should_eval:
                student_name = self.cfg["model"].get("student", "student")
                self.logger.info(f"Evaluating student ({student_name})...")
                eval_results = run_student_evaluation(
                    self.distiller.student,
                    self.val_loader,
                    self.device,
                    self.cfg["model"]["num_classes"],
                    student_name=student_name,
                )
                miou = eval_results["mIoU"]
                self.logger.info(
                    f"Student mIoU: {miou:.2f}, FLOPs: {eval_results['FLOPs']:.2f}G, "
                    f"Params: {eval_results['Params']:.2f}M, FPS: {eval_results['FPS']:.2f}"
                )
                if miou > best_miou:
                    best_miou = miou
                    best_path = os.path.join(
                        self.output_dir, f"{self.cfg['experiment']['name']}_best.pth"
                    )
                    torch.save(self.distiller.student.state_dict(), best_path)
                    self.logger.info(f"New best mIoU {best_miou:.2f} -> {best_path}")

            if (global_iter + 1) % self.eval_interval == 0:
                ckpt_path = os.path.join(
                    self.output_dir,
                    f"{self.cfg['experiment']['name']}_iter_{global_iter + 1}.pth",
                )
                torch.save(
                    {
                        "iter": global_iter + 1,
                        "epoch": epoch,
                        "model_state_dict": self.distiller.student.state_dict(),
                        "distiller_state_dict": self.distiller.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    ckpt_path,
                )

            global_iter += 1

        self.logger.info(f"Training finished. Best mIoU: {best_miou:.2f}")
