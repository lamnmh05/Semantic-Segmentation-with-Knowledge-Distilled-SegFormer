import time

import numpy as np
import torch
import torch.nn.functional as F

from fvcore.nn import FlopCountAnalysis, parameter_count
from thop import profile as thop_profile


def fast_hist(gt, pred, num_classes):
    k = (gt >= 0) & (gt < num_classes)
    return np.bincount(
        num_classes * gt[k].astype(int) + pred[k].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def per_class_iou(hist):
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)
    return np.nan_to_num(ious)


class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def update(self, pred, gt):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        mask = gt != 255
        self.hist += fast_hist(gt[mask].flatten(), pred[mask].flatten(), self.num_classes)

    def compute_miou(self):
        ious = per_class_iou(self.hist)
        return float(np.nanmean(ious) * 100), ious * 100


@torch.no_grad()
def student_predict(student, images):
    logits = student(images)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits


def evaluate_student(student, dataloader, device, num_classes):
    student.eval()
    evaluator = Evaluator(num_classes)

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = student_predict(student, images)
            logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            preds = torch.argmax(logits, dim=1)
            evaluator.update(preds, labels)

            if (i + 1) % 50 == 0:
                print(f"Student eval: {i + 1}/{len(dataloader)}")

    miou, _ = evaluator.compute_miou()
    return miou


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def get_flops_params(model, device, input_size=(1, 3, 512, 512)):
    model.eval()
    dummy_input = torch.randn(input_size, device=device)

    if FlopCountAnalysis is not None:
        flops = FlopCountAnalysis(model, dummy_input).total()
        params = parameter_count(model)[""]
        return flops / 1e9, params / 1e6

    if thop_profile is not None:
        macs, params = thop_profile(model, inputs=(dummy_input,), verbose=False)
        return (macs * 2) / 1e9, params / 1e6

    return 0.0, count_parameters(model)


def measure_fps(model, device, input_size=(1, 3, 512, 512), num_iterations=100):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            student_predict(model, dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            student_predict(model, dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return num_iterations / (time.time() - start_time)


def run_student_evaluation(student, dataloader, device, num_classes, student_name="student"):
    was_training = student.training
    student.eval()

    miou = evaluate_student(student, dataloader, device, num_classes)
    flops, params = get_flops_params(student, device)
    fps = measure_fps(student, device)

    print("-" * 50)
    print(f"Student evaluation ({student_name})")
    print(f"{'':12} | FLOPs(G) | Param(M) | FPS    | mIoU")
    print(f"{student_name:12} | {flops:8.2f} | {params:8.2f} | {fps:6.2f} | {miou:5.2f}")
    print("-" * 50)

    if was_training:
        student.train()

    return {"mIoU": miou, "FLOPs": flops, "Params": params, "FPS": fps}


run_full_evaluation = run_student_evaluation
