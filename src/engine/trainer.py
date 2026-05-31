import os
import logging
import time
import datetime

import torch
from tqdm.auto import tqdm

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
        self.lr_schedule = train_cfg.get("lr_schedule", "poly")
        self.step_epochs = train_cfg.get("step_epochs", [16, 22])
        self.step_gamma = train_cfg.get("step_gamma", 0.1)
        self.base_lr = train_cfg["lr"]
        self.grad_clip = train_cfg.get("grad_clip")

        distill_cfg = cfg.get("distill", {})
        self.method = distill_cfg.get("method")
        self.connector_warmup_iters = 0
        if self.method == "AttnFD":
            if train_cfg.get("connector_warmup_iters") is not None:
                self.connector_warmup_iters = int(train_cfg["connector_warmup_iters"])
            else:
                warmup_epochs = train_cfg.get("connector_warmup_epochs", 0)
                self.connector_warmup_iters = warmup_epochs * len(train_loader)
        self.hint_pretrain_iters = 0
        if self.method == "FitNet":
            self.hint_pretrain_iters = distill_cfg.get("hint_pretrain_iters", 0)

        self.output_dir = cfg["experiment"]["output_dir"]
        self.logger = setup_logger(cfg["experiment"]["log_dir"], cfg["experiment"]["name"])
        os.makedirs(self.output_dir, exist_ok=True)

        img_size = cfg["dataset"]["img_size"]
        if isinstance(img_size, (list, tuple)):
            h, w = img_size[0], img_size[1]
        else:
            h = w = img_size
        self.eval_input_size = (1, 3, h, w)
        self.loss_history = []

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
        if hasattr(self.distiller, "get_regressor_parameters"):
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
        early_stopping_counter = 0
        early_stopping_patience = 10
        global_iter = 0
        epoch = 0
        train_iter = iter(self.train_loader)
        start_time = time.time()
        pbar = tqdm(total=self.max_iters, initial=global_iter, desc="Training", dynamic_ncols=True)

        try:
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

                current_epoch_val = global_iter / max(1, len(self.train_loader))
                lr = get_lr(
                    global_iter, self.base_lr, self.max_iters,
                    warmup_iters=self.warmup_iters,
                    warmup_ratio=self.warmup_ratio,
                    power=self.poly_power,
                    schedule=self.lr_schedule,
                    current_epoch=current_epoch_val,
                    step_epochs=self.step_epochs,
                    step_gamma=self.step_gamma
                )
                set_optimizer_lr(self.optimizer, lr)

                self.optimizer.zero_grad()
                hint_only = self._is_hint_pretrain(global_iter)
                stage = "hint" if hint_only else "kd"
                logits, losses = self.distiller.forward_train(
                    images, targets, hint_only=hint_only, stage=stage
                )
                loss = losses["loss_total"]
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.distiller.parameters() if p.requires_grad],
                        self.grad_clip,
                    )
                self.optimizer.step()

                # Record loss history
                scalar_losses = {
                    name: value.item() if torch.is_tensor(value) else float(value)
                    for name, value in losses.items()
                }
                self.loss_history.append({
                    "iter": global_iter + 1,
                    "epoch": (global_iter + 1) / len(self.train_loader),
                    "lr": lr,
                    **scalar_losses,
                })

                phase = "warmup" if self._is_connector_warmup(global_iter) else (
                    "hint" if hint_only else "train"
                )
                pbar.set_postfix(
                    loss=float(loss.item()),
                    ce=float(scalar_losses.get("loss_ce", 0.0)),
                    kd=float(scalar_losses.get("loss_kd", 0.0)),
                    lr=f"{lr:.2e}",
                    phase=phase,
                )
                pbar.update(1)

                if (global_iter + 1) % self.log_interval == 0:
                    elapsed = time.time() - start_time
                    time_per_iter = elapsed / (global_iter + 1)
                    eta_seconds = time_per_iter * (self.max_iters - global_iter - 1)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    total_epochs = self.cfg["train"].get("epochs", 50)
                    fractional_epoch = (global_iter + 1) / len(self.train_loader)
                    loss_summary = " ".join(
                        f"{name}={value:.4f}" for name, value in scalar_losses.items()
                    )
                    self.logger.info(
                        f"Iter [{global_iter + 1}/{self.max_iters}] epoch=[{fractional_epoch:.2f}/{total_epochs}] phase={phase} "
                        f"lr={lr:.2e} {loss_summary} "
                        f"time/iter={time_per_iter:.3f}s eta={eta_string}"
                    )

                at_eval_step = self.val_loader and (global_iter + 1) % self.eval_interval == 0
                in_warmup = self._is_connector_warmup(global_iter) or self._is_hint_pretrain(global_iter)
                if at_eval_step and in_warmup:
                    self.logger.info(
                        f"Iter [{global_iter + 1}] skip eval (connector/hint warmup)"
                    )
                if at_eval_step and not in_warmup:
                    student_name = self.cfg["model"].get("student", "student")
                    self.logger.info(f"Evaluating student ({student_name})...")
                    eval_results = run_student_evaluation(
                        self.distiller.student,
                        self.val_loader,
                        self.device,
                        self.cfg["model"]["num_classes"],
                        student_name=student_name,
                        input_size=self.eval_input_size,
                    )
                    miou = eval_results["mIoU"]
                    self.logger.info(
                        f"Student mIoU: {miou:.2f}, FLOPs: {eval_results['FLOPs']:.2f}G, "
                        f"Params: {eval_results['Params']:.2f}M, FPS: {eval_results['FPS']:.2f}"
                    )
                    if miou > best_miou:
                        best_miou = miou
                        early_stopping_counter = 0
                        best_path = os.path.join(
                            self.output_dir, f"{self.cfg['experiment']['name']}_best.pth"
                        )
                        torch.save(self.distiller.student.state_dict(), best_path)
                        self.logger.info(f"New best mIoU {best_miou:.2f} -> {best_path}")
                    else:
                        early_stopping_counter += 1
                        self.logger.info(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                        if early_stopping_counter >= early_stopping_patience:
                            self.logger.info("Early stopping triggered. Stopping training.")
                            break

                # Checkpoint saving per interval is disabled to prevent disk overflow as requested.
                # Only the best checkpoint (*_best.pth) is saved during evaluation.
                pass

                global_iter += 1
        finally:
            pbar.close()

        # Save loss history at the end of training
        loss_path = os.path.join(self.output_dir, f"{self.cfg['experiment']['name']}_loss.json")
        import json
        with open(loss_path, "w", encoding="utf-8") as f:
            json.dump(self.loss_history, f, indent=4)
        self.logger.info(f"Saved loss history to {loss_path}")

        self.logger.info(f"Training finished. Best mIoU: {best_miou:.2f}")
