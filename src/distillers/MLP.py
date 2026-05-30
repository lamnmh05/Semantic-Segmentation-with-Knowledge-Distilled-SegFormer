import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPFDConnector(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # MLP Channel-wise: C_s -> hidden_channels -> C_t
        # Sử dụng 1x1 Conv2d tương đương với Linear layer áp dụng lên từng pixel
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        )
        
        # Khởi tạo trọng số 
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, student_feat):
        """
        student_feat: Tensor kích thước (B, C_s, H, W)
        return: Tensor kích thước (B, C_t, H, W)
        """
        return self.mlp(student_feat)
class MLPFDistiller(nn.Module):
    def __init__(self, student_model, teacher_model, config):
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        
        # Đóng băng toàn bộ trọng số của Teacher 
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Đọc config để lấy trọng số loss và layer cần distill
        self.ce_weight = config['distill']['ce_weight']
        self.feat_weight = config['distill']['feat_weight']
        self.feat_layer = config['distill']['feat_layer'] 
        
        # Khởi tạo Connector 
        in_channels = config['distill']['student_channels'][self.feat_layer]
        out_channels = config['distill']['teacher_channels'][self.feat_layer]
        hidden_channels = config['distill']['mlp_hidden_channel']
        
        self.connector = MLPFDConnector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels
        )

    def forward(self, pixel_values, labels=None):
        # 1. Chạy Teacher 
        with torch.no_grad():
            self.teacher.eval() # Đảm bảo luôn ở trạng thái eval
            teacher_outputs = self.teacher(
                pixel_values=pixel_values, 
                output_hidden_states=True
            )
            # Lấy feature map của Teacher ở Stage 4
            t_feat = teacher_outputs.hidden_states[self.feat_layer]

        # 2. Chạy Student
        student_outputs = self.student(
            pixel_values=pixel_values, 
            labels=labels, 
            output_hidden_states=True
        )
        # Lấy feature map của Student ở Stage 4
        s_feat = student_outputs.hidden_states[self.feat_layer]

        # 3. Đi qua MLP Connector để scale channel của Student lên bằng Teacher
        s_mapped = self.connector(s_feat)

        # 4. Tính Feature Distillation Loss (MSE)
        # Dùng hàm F.mse_loss đã được import ở đầu file
        loss_fd = F.mse_loss(s_mapped, t_feat)

        # 5. Tính Tổng Loss
        loss_ce = student_outputs.loss 
        
        total_loss = (self.ce_weight * loss_ce) + (self.feat_weight * loss_fd)

        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_fd": loss_fd,
            "logits": student_outputs.logits
        }