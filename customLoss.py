import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class VGGLoss(nn.Module):
    def __init__(self, device, feature_layer=35):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features[:feature_layer+1].eval()
        self.vgg = nn.Sequential(*list(vgg.children())[:feature_layer+1]).to(device)
        # 注册归一化参数（保持3通道）
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device))
        
    def preprocess_vgg(self, x):
        """将阴影信息融合到RGB生成3通道"""
        rgb = x[:, :3]  # [B,3,H,W]
        shadow = x[:, 3:]  # [B,1,H,W]
        # 确保数据类型一致
        shadow = shadow.to(rgb.dtype)
        return rgb * (1 - shadow)  # 阴影区域变暗

    def forward(self, output, target, inputs):
        # 确保输入类型一致
        output = output.to(torch.float32)
        target = target.to(torch.float32)
        inputs = inputs.to(torch.float32)
        
        # 预处理（生成3通道）
        output_rgb = self.preprocess_vgg(torch.cat([inputs[:, :3], output], dim=1))
        target_rgb = self.preprocess_vgg(torch.cat([inputs[:, :3], target], dim=1))
        
        # 归一化
        output_norm = (output_rgb - self.mean) / self.std
        target_norm = (target_rgb - self.mean) / self.std
        
        # 提取特征
        output_feat = self.vgg(output_norm)
        target_feat = self.vgg(target_norm)
        
        return F.mse_loss(output_feat, target_feat)

class CustomLoss(nn.Module):
    def __init__(self, device, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.vgg_loss = VGGLoss(device)
        
    def forward(self, output, target, inputs):
        # 确保输出经过Sigmoid
        assert output.min() >= 0 and output.max() <= 1, "输出必须经过Sigmoid激活!"
        
        # 计算L1损失
        l1 = self.l1(output, target)
        
        # 计算VGG损失
        vgg = self.vgg_loss(output, target, inputs)
        
        # 组合损失
        return self.alpha * l1 + (1 - self.alpha) * vgg