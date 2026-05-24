"""Iteration-based LR schedules aligned with SegFormer (poly + warmup)."""


def poly_lr(base_lr, iter_idx, max_iters, power=1.0):
    return base_lr * (1 - iter_idx / max(1, max_iters)) ** power


def warmup_lr(base_lr, iter_idx, warmup_iters, warmup_ratio):
    if warmup_iters <= 0:
        return base_lr
    if iter_idx >= warmup_iters:
        return base_lr
    return base_lr * (warmup_ratio + (1.0 - warmup_ratio) * (iter_idx / warmup_iters))


def get_lr(iter_idx, base_lr, max_iters, warmup_iters=0, warmup_ratio=1e-6, power=1.0):
    lr = poly_lr(base_lr, iter_idx, max_iters, power=power)
    if iter_idx < warmup_iters:
        lr = warmup_lr(base_lr, iter_idx, warmup_iters, warmup_ratio)
    return lr


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_mult", 1.0)
