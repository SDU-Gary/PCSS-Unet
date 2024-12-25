import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.checkpoint import checkpoint
from torchvision.models import VGG19_Weights

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.9):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()  # 使用L1损失代替BCE
        
        # 加载预训练的VGG19用于特征提取
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # 启用VGG的gradient checkpointing
        vgg.features.apply(lambda m: m.register_forward_hook(lambda m, _, __: setattr(m, 'grad_checkpointing', True)))
        self.vgg = vgg.features.eval()
        # 冻结VGG19的参数
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def get_features(self, x):
        features = []
        current = x
        
        def custom_forward(start, end):
            def forward(*inputs):
                x = inputs[0]
                for layer in self.vgg[start:end]:
                    x = layer(x)
                return x
            return forward
        
        # 将VGG分成几个块来使用gradient checkpointing
        block_size = 4  # 每4层作为一个块
        for i in range(0, len(self.vgg), block_size):
            end_idx = min(i + block_size, len(self.vgg))
            current = checkpoint(custom_forward(i, end_idx), current)
            if isinstance(self.vgg[end_idx-1], nn.ReLU):
                features.append(current)
                
        return features
    
    def forward(self, output, target, inputs):
        """
        output: 网络输出 [batch_size, 1, H, W]
        target: 目标图像 [batch_size, 1, H, W]
        inputs: 原始输入图像 [batch_size, 4, H, W]
        """
        assert output.size() == target.size(), "输出和目标尺寸不匹配"
        assert inputs.size(2) == target.size(2) and inputs.size(3) == target.size(3), "输入和目标的高度和宽度不匹配"
        assert inputs.size(1) == 4, "输入应该是4通道"
        assert output.size(1) == 1 and target.size(1) == 1, "输出和目标应该是单通道"

        # L1损失
        l1_loss = self.l1(output, target)

        # 如果是单通道图像，需要复制到3通道以适应VGG
        output_3c = output.repeat(1, 3, 1, 1)
        target_3c = target.repeat(1, 3, 1, 1)
        
        # VGG特征损失
        output_features = self.get_features(output_3c)
        target_features = self.get_features(target_3c)
        
        # 计算VGG特征损失
        vgg_loss = 0
        for out_feat, target_feat in zip(output_features, target_features):
            vgg_loss += F.mse_loss(out_feat, target_feat)
            
        # 按照论文公式组合损失
        total_loss = self.alpha * l1_loss + (1 - self.alpha) * vgg_loss
        
        return total_loss