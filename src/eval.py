import torch
import torch.nn.functional as F
import numpy as np
import time

def fast_hist(a, b, n):
    """
    Return histogram for evaluation
    a: Prediction
    b: Ground truth
    n: Number of classes
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n ** 2).reshape(n, n)

def per_class_iou(hist):
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)
    return np.nan_to_num(ious)

class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def update(self, pred, label):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
            
        mask = (label != 255)
        pred = pred[mask]
        label = label[mask]
        
        self.hist += fast_hist(label.flatten(), pred.flatten(), self.num_classes)

    def compute_miou(self):
        ious = per_class_iou(self.hist)
        miou = np.nanmean(ious) * 100
        return miou, ious * 100

def get_flops_params(model, device, input_size=(1, 3, 512, 512)):
    """Calculate FLOPs (G) and Params (M)"""
    dummy_input = torch.randn(input_size).to(device)
    
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        model.eval()
        flops = FlopCountAnalysis(model, dummy_input).total()
        params = parameter_count(model)[""]
        return flops / 1e9, params / 1e6
    except ImportError:
        pass
    
    try:
        from thop import profile
        model.eval()
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        return (macs * 2) / 1e9, params / 1e6
    except ImportError:
        pass

    params = sum(p.numel() for p in model.parameters())
    print("Warning: fvcore or thop not found, returning 0 for FLOPs.")
    return 0.0, params / 1e6

def measure_fps(model, device, input_size=(1, 3, 512, 512), num_iterations=100):
    """Measure inference FPS"""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    total_time = time.time() - start_time
    fps = num_iterations / total_time
    return fps

def evaluate_model(model, dataloader, device, num_classes):
    print("Starting evaluation...")
    model.eval()
    evaluator = Evaluator(num_classes)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            try:
                _, _, logits = model.extract_feature(images)
            except:
                logits = model(images)
                
            logits = F.interpolate(logits, size=labels.shape[1:], mode='bilinear', align_corners=False)
            preds = torch.argmax(logits, dim=1)
            
            evaluator.update(preds, labels)
            
            if (i + 1) % 50 == 0:
                print(f"Eval Progress: {i+1}/{len(dataloader)}")
                
    miou, class_ious = evaluator.compute_miou()
    return miou

def run_full_evaluation(model, dataloader, device, num_classes):
    miou = evaluate_model(model, dataloader, device, num_classes)
    flops, params = get_flops_params(model, device)
    fps = measure_fps(model, device)
    
    print("-" * 50)
    print(f"Evaluation Results:")
    print(f"Method | FLOPs(G) | Param(M) | FPS(S) | mIoU")
    print(f"Ours   | {flops:8.2f} | {params:8.2f} | {fps:6.2f} | {miou:5.2f}")
    print("-" * 50)
    
    return {
        "mIoU": miou,
        "FLOPs": flops,
        "Params": params,
        "FPS": fps
    }
