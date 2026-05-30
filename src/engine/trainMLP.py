import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from src.distillers.MLP import MLPFDistiller
from src.dataloaders.ade20k import get_ade20k_dataloader

def load_config(config_path="configs/MLP_ADE20K.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def intersect_and_union(pred, label, num_classes, ignore_index=255):
    mask = (label != ignore_index)
    pred = pred[mask]
    label = label[mask]
    
    # Tính toán tọa độ giao thoa
    intersect = np.bincount(
        label * num_classes + pred, 
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    
    return np.diag(intersect), intersect.sum(axis=0), intersect.sum(axis=1)

def evaluate_model(model, val_loader, device, num_classes, ignore_index):
    """
    Hàm chạy Validation để tính toán chỉ số mIoU trong lúc train
    """
    model.eval() # Chuyển Student sang chế độ eval
    total_intersect = np.zeros((num_classes,), dtype=float)
    total_union = np.zeros((num_classes,), dtype=float)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="📊 Đang đánh giá mIoU", leave=False):
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward qua mô hình Student độc lập
            logits = model(pixel_values=images).logits
            
            upsampled_logits = F.interpolate(
                logits, 
                size=labels.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            preds = upsampled_logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Tích lũy kết quả các batch
            for i in range(images.size(0)):
                intersect, union, _ = intersect_and_union(preds[i], labels_np[i], num_classes, ignore_index)
                total_intersect += intersect
                total_union += union

    # Tính mIoU trung bình trên các class hợp lệ
    valid_classes = total_union > 0
    iou = total_intersect[valid_classes] / total_union[valid_classes]
    mIoU = np.mean(iou) * 100
    
    return mIoU

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config['experiment']['output_dir'], exist_ok=True)

    num_classes = config['model']['num_classes']
    ignore_index = config['distill']['ignore_index']

    # 2. Khởi tạo Teacher Model (SegFormer-B4)
    teacher = SegformerForSemanticSegmentation.from_pretrained(
        config['model']['teacher'],
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    # Đóng băng hoàn toàn Teacher
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    # 3. Khởi tạo Student Model (SegFormer-B0)
    student_config = SegformerConfig.from_pretrained("nvidia/mit-b0")
    student_config.num_labels = num_classes
    student = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", 
        config=student_config, 
        ignore_mismatched_sizes=True
    )

    # 4. Gắn vào Bộ điều phối Distiller
    distiller = MLPFDistiller(student, teacher, config).to(device)

    # 5. Train Loader & Val Loader
    train_loader = get_ade20k_dataloader(
        data_root=config['dataset']['data_root'],
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        img_size=config['dataset']['img_size'],
        split='train',
        augment=config['dataset']['augment']
    )

    val_loader = get_ade20k_dataloader(
        data_root=config['dataset']['data_root'],
        batch_size=1, # Eval bắt buộc dùng batch size = 1 để chính xác nhất
        num_workers=config['dataset']['num_workers'],
        img_size=config['dataset']['img_size'],
        split='val',
        augment=False 
    )

    # 6. Thiết lập Optimizer với mức Learning Rate phân tầng
    base_lr = config['train']['lr']
    head_lr_mult = config['train']['head_lr_mult']
    connector_lr_mult = config['train']['connector_lr_mult']

    param_groups = [
        {'params': distiller.student.segformer.parameters(), 'lr': base_lr},
        {'params': distiller.student.decode_head.parameters(), 'lr': base_lr * head_lr_mult},
        {'params': distiller.connector.parameters(), 'lr': base_lr * connector_lr_mult},
    ]

    optimizer = AdamW(
        param_groups, 
        weight_decay=config['train']['weight_decay'], 
        betas=tuple(config['train']['betas'])
    )

    # Scheduler PolyLR theo số lượng Iterations
    scheduler = PolynomialLR(
        optimizer, 
        total_iters=config['train']['max_iters'], 
        power=config['train']['poly_power']
    )

    # Quản lý tăng tốc độ tính toán phần cứng bằng FP16
    scaler = GradScaler()

    # 7. VÒNG LẶP HUẤN LUYỆN CHÍNH
    
    current_iter = 0
    max_iters = config['train']['max_iters']
    log_interval = config['train']['log_interval']
    eval_interval = config['train']['eval_interval']
    best_miou = 0.0  # Biến lưu kỷ lục mIoU cao nhất đạt được

    pbar = tqdm(total=max_iters, desc="Huấn luyện")

    while current_iter < max_iters:
        distiller.train()        # Chuyển Student + Connector về mode Train
        distiller.teacher.eval() # Giữ Teacher luôn ở mode Eval
        
        for batch in train_loader:
            if current_iter >= max_iters:
                break

            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = distiller(pixel_values, labels)
                loss = outputs['loss']
                loss_ce = outputs['loss_ce']
                loss_fd = outputs['loss_fd']

            # Cập nhật trọng số mạng thông qua GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            # Xuất nhật ký hiển thị nhanh trên thanh tiến trình (Log Bar)
            if current_iter % log_interval == 0:
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'CE': f"{loss_ce.item():.4f}",
                    'FD': f"{loss_fd.item():.4f}",
                    'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
                })

            current_iter += 1
            pbar.update(1)

            # --- EVALUATION INTERVAl ---
            if current_iter % eval_interval == 0:
                print(f"\n [Iter {current_iter}] Validation...")
                
                # Gọi hàm eval tính toán mIoU thực tế
                current_miou = evaluate_model(
                    distiller.student, val_loader, device, num_classes, ignore_index
                )
                
                print(f"  mIoU tại Iter {current_iter} là: {current_miou:.2f}%")
                
                # Best Checkpoint
                if current_miou > best_miou:
                    best_miou = current_miou
                    save_path = os.path.join(config['experiment']['output_dir'], "student_best.pth")
                    torch.save(distiller.student.state_dict(), save_path)
                    print(f"Đã cập nhật file tốt nhất: {save_path} (Best mIoU: {best_miou:.2f}%)")
                else:
                    print(f" Chưa vượt qua kỷ lục cũ trước đây ({best_miou:.2f}%)")
                
                # Trả lại trạng thái phân phối ban đầu cho bộ distiller để tiếp tục vòng lặp train
                distiller.train()
                distiller.teacher.eval()

    pbar.close()
    print(f"  Kết quả mIoU đạt được cuối cùng: {best_miou:.2f}%")

if __name__ == "__main__":
    main()