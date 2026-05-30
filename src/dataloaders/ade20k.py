import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ADE20KDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=(512, 512), augment=True):
        """
        data_root: Đường dẫn tới thư mục ADEChallengeData2016
        split: 'train' hoặc 'val'
        """
        super().__init__()
        self.data_root = data_root
        self.split = 'training' if split == 'train' else 'validation'
        
        # Đường dẫn tới thư mục images và annotations
        self.img_dir = os.path.join(data_root, 'images', self.split)
        self.ann_dir = os.path.join(data_root, 'annotations', self.split)
        
        # Lấy danh sách tất cả các file ảnh (.jpg)
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
        # Khởi tạo Augmentation Pipeline
        self.transform = self._get_transforms(img_size, split, augment)

    def _get_transforms(self, img_size, split, augment):
        # Resize cơ bản và chuẩn hóa (Normalize theo chuẩn ImageNet cho Segformer)
        base_transforms = [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
        
        if split == 'train' and augment:
            # Thêm các phép biến đổi data augmentation cho tập train
            train_transforms = [
                A.RandomScale(scale_limit=(-0.5, 1.0), p=0.5), # Scale range [0.5, 2.0]
                A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], 
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.RandomCrop(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
            ]
            # Albumentations chạy theo thứ tự: Augment -> Resize/Crop -> Normalize -> ToTensor
            return A.Compose(train_transforms + base_transforms)
        
        return A.Compose(base_transforms)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        
        # Đọc ảnh gốc (RGB)
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Đọc mask label (Grayscale)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.ann_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Thực hiện phép biến đổi (Resize, Flip, Scale...)
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        mask = mask.to(torch.long)
        mask = mask - 1
        mask[mask == -1] = 255 # Hoặc config['distill']['ignore_index']
        
        return {
            "pixel_values": image, # Shape: (3, H, W), Dtype: float32
            "labels": mask         # Shape: (H, W), Dtype: long
        }

def get_ade20k_dataloader(data_root, batch_size=2, num_workers=4, img_size=(512, 512), split='train', augment=True):
    dataset = ADE20KDataset(
        data_root=data_root, 
        split=split, 
        img_size=img_size, 
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,     
        drop_last=(split == 'train') 
    )
    
    return dataloader