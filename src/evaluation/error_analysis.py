import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class ErrorAnalyzer:
    def __init__(self, num_classes, output_dir, label_names=None):
        self.num_classes = num_classes
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.label_names = label_names if label_names else [f"Class {i}" for i in range(num_classes)]
        
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
        self.colors[255] = [0, 0, 0]

    def save_per_class_metrics(self, ious, miou):
        csv_path = os.path.join(self.output_dir, "per_class_metrics.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Class_ID,Class_Name,IoU\n")
            for i in range(self.num_classes):
                f.write(f"{i},{self.label_names[i]},{ious[i]:.2f}\n")
            f.write(f"Total,mIoU,{miou:.2f}\n")
        print(f"Saved per-class IoU at {csv_path}")

    def save_confusion_matrix(self, hist):
        np.save(os.path.join(self.output_dir, "confusion_matrix.npy"), hist)
        
        row_sums = hist.sum(axis=1, keepdims=True)
        norm_hist = np.divide(hist, row_sums, out=np.zeros_like(hist), where=row_sums!=0)
        
        plt.figure(figsize=(20, 20))
        plt.imshow(norm_hist, cmap='Blues')
        plt.colorbar()
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"), dpi=300)
        plt.close()
        print(f"Saved Confusion Matrix at {self.output_dir}")

    def decode_segmap(self, mask):
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        for l in range(0, self.num_classes):
            idx = mask == l
            r[idx] = self.colors[l, 0]
            g[idx] = self.colors[l, 1]
            b[idx] = self.colors[l, 2]
            
        idx = mask == 255
        r[idx] = 0
        g[idx] = 0
        b[idx] = 0
        return np.stack([r, g, b], axis=2)

    def save_visualizations(self, image_tensor, gt_tensor, pred_tensor, filename):
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
        img = image_tensor[0].cpu().numpy()
        img = (img * std + mean) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        
        gt = gt_tensor[0].cpu().numpy()
        pred = pred_tensor[0].cpu().numpy()
        
        gt_color = self.decode_segmap(gt)
        pred_color = self.decode_segmap(pred)
        
        overlay_pred = cv2.addWeighted(img, 0.6, pred_color, 0.4, 0)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(gt_color)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')

        axes[2].imshow(pred_color)
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')
        
        axes[3].imshow(overlay_pred)
        axes[3].set_title("Overlay Prediction")
        axes[3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"{filename}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
