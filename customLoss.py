import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.9, p=3):
        super(CustomLoss, self).__init__()
        self.alpha = alpha        # 权重系数
        self.p = p               # 扰动数量
        self.bce = nn.BCEWithLogitsLoss()  # 基础的二元交叉熵损失
        
        # 加载预训练的VGG19用于特征提取
        self.vgg = models.vgg19(pretrained=True).features.eval()
        # 冻结VGG19的参数
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def get_features(self, x):
        # 提取VGG19特征
        features = []
        for i, layer in enumerate(self.vgg.children()):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features
    
    def forward(self, output, target, inputs):
        """
        output: 网络输出 [batch_size, 1, H, W]
        target: 目标图像 [batch_size, 1, H, W]
        inputs: 原始输入图像（包含扰动）[batch_size, 4, H, W]
        """
        assert output.size() == target.size(), "输出和目标尺寸不匹配"
        assert inputs.size(2) == target.size(2) and inputs.size(3) == target.size(3), "输入和目标的高度和宽度不匹配"
        assert inputs.size(1) == 4, "输入应该是4通道"
        assert output.size(1) == 1 and target.size(1) == 1, "输出和目标应该是单通道"

        # 主损失：网络输出与目标之间的BCE损失
        main_loss = self.bce(output, target)

        # 从输入中选择RGB通道用于特征提取
        inputs_rgb = inputs[:, :3, :, :]  # 只取前3个通道作为RGB

        # 如果是单通道图像，需要复制到3通道以适应VGG
        output_3c = output.repeat(1, 3, 1, 1)
        target_3c = target.repeat(1, 3, 1, 1)
        
        # VGG特征损失
        output_features = self.get_features(output_3c)
        target_features = self.get_features(target_3c)
        input_features = self.get_features(inputs_rgb)
        
        # 计算VGG特征损失
        vgg_loss = 0
        for out_feat, target_feat in zip(output_features, target_features):
            vgg_loss += F.mse_loss(out_feat, target_feat)
            
        # 扰动损失：比较输出特征与输入特征的差异
        perturbation_loss = 0
        batch_size = inputs.size(0)
        for out_feat, input_feat in zip(output_features, input_features):
            # 对每个batch样本计算损失
            for i in range(batch_size):
                perturbation_loss += F.mse_loss(out_feat[i:i+1], input_feat[i:i+1])
        perturbation_loss = perturbation_loss / (len(output_features) * batch_size)  # 归一化
        
        # 总损失
        total_loss = self.alpha * (main_loss + vgg_loss) + (1 - self.alpha) * perturbation_loss
        
        return total_loss