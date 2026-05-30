import os
import yaml
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from src.distillers.MLP import MLPFDistiller
from src.dataloaders.ade20k import get_ade20k_dataloader  

def load_config(config_path="configs/mlp_ade20k.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # 1. Load Cấu hình
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Bắt đầu quá trình training trên thiết bị: {device}")

    # Tạo thư mục lưu checkpoint
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)

    # 2. Khởi tạo Teacher Model (SegFormer-B4)
    teacher = SegformerForSemanticSegmentation.from_pretrained(
        config['model']['teacher'],
        num_labels=config['model']['num_classes'],
        ignore_mismatched_sizes=True
    )
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    # 3. Khởi tạo Student Model (SegFormer-B0)
    student_config = SegformerConfig.from_pretrained("nvidia/mit-b0")
    student_config.num_labels = config['model']['num_classes']
    student = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", 
        config=student_config, 
        ignore_mismatched_sizes=True
    )

    # 4. Gắn Teacher và Student vào Distiller (Bộ điều phối)
    distiller = MLPFDistiller(student, teacher, config).to(device)

    # 5. Load Dữ liệu (ADE20K)
    train_loader = get_ade20k_dataloader(
        data_root=config['dataset']['data_root'],
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        img_size=config['dataset']['img_size'],
        split='train'
    )

    # 6. Cấu hình Optimizer với Learning Rate Multiplier
    # (Decode head và Connector cần LR cao hơn Encoder)
    base_lr = config['train']['lr']
    head_lr_mult = config['train']['head_lr_mult']
    connector_lr_mult = config['train']['connector_lr_mult']

    param_groups = [
        # Nhóm 1: Student Encoder (Base LR)
        {'params': distiller.student.segformer.parameters(), 'lr': base_lr},
        # Nhóm 2: Student Decode Head (Base LR * 10)
        {'params': distiller.student.decode_head.parameters(), 'lr': base_lr * head_lr_mult},
        # Nhóm 3: MLP Connector (Base LR * 10)
        {'params': distiller.connector.parameters(), 'lr': base_lr * connector_lr_mult},
    ]

    optimizer = AdamW(
        param_groups, 
        weight_decay=config['train']['weight_decay'], 
        betas=tuple(config['train']['betas'])
    )

    # Scheduler (PolyLR thường dùng trong Segmentation)
    scheduler = PolynomialLR(
        optimizer, 
        total_iters=config['train']['max_iters'], 
        power=config['train']['poly_power']
    )

    scaler = GradScaler()

    # 7.Training Loop
    print(" Bắt đầu huấn luyện...")
    distiller.train()
    distiller.teacher.eval() 

    current_iter = 0
    max_iters = config['train']['max_iters']
    log_interval = config['train']['log_interval']

    pbar = tqdm(total=max_iters, desc="Huấn luyện")

    while current_iter < max_iters:
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

            # Backward và Optimizer step với GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            # Logging
            if current_iter % log_interval == 0:
                pbar.set_postfix({
                    'Total Loss': f"{loss.item():.4f}",
                    'CE Loss': f"{loss_ce.item():.4f}",
                    'FD Loss': f"{loss_fd.item():.4f}",
                    'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
                })

            current_iter += 1
            pbar.update(1)

            # Lưu Checkpoint định kỳ (dựa vào eval_interval)
            if current_iter % config['train']['eval_interval'] == 0:
                save_path = os.path.join(config['experiment']['output_dir'], f"student_iter_{current_iter}.pth")
                torch.save(distiller.student.state_dict(), save_path)
                print(f"\n Đã lưu checkpoint tại {save_path}")

    pbar.close()
    print(" Hoàn tất huấn luyện!")

if __name__ == "__main__":
    main()