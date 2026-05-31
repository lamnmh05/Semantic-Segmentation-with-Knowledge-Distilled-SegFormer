"""Iteration-based LR schedules aligned with SegFormer (poly + warmup)."""


def poly_lr(base_lr, iter_idx, max_iters, power=1.0):
    return base_lr * (1 - iter_idx / max(1, max_iters)) ** power


def step_lr(base_lr, current_epoch, step_epochs, gamma=0.1):
    lr = base_lr
    for step_epoch in step_epochs:
        if current_epoch >= step_epoch:
            lr *= gamma
    return lr


def warmup_lr(base_lr, iter_idx, warmup_iters, warmup_ratio):
    if warmup_iters <= 0:
        return base_lr
    if iter_idx >= warmup_iters:
        return base_lr
    return base_lr * (warmup_ratio + (1.0 - warmup_ratio) * (iter_idx / warmup_iters))


def get_lr(iter_idx, base_lr, max_iters, warmup_iters=0, warmup_ratio=1e-6, power=1.0,
           schedule="poly", current_epoch=0, step_epochs=None, step_gamma=0.1):
    if schedule == "step":
        if step_epochs is None:
            step_epochs = [16, 22]
        lr = step_lr(base_lr, current_epoch, step_epochs, gamma=step_gamma)
    else:
        lr = poly_lr(base_lr, iter_idx, max_iters, power=power)
        
    if iter_idx < warmup_iters:
        lr = warmup_lr(base_lr, iter_idx, warmup_iters, warmup_ratio)
    return lr


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_mult", 1.0)
