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
        for layer in self.vgg:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features
    
    def forward(self, output, target, inputs):
        """
        output: 网络输出
        target: 目标图像
        inputs: 原始输入图像（包含扰动）
        """
        assert output.size() == target.size(), "输出和目标尺寸不匹配"
        assert inputs.size() == target.size(), "输入和目标尺寸不匹配"

        # 主损失：网络输出与目标之间的BCE损失
        main_loss = self.bce(output, target)

        # 如果是单通道图像，需要复制到3通道以适应VGG
        if output.size(1) == 1:
            output = output.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            inputs = inputs.repeat(1, 3, 1, 1)
        
        # VGG特征损失
        output_features = self.get_features(output)
        target_features = self.get_features(target)
        vgg_loss = 0
        for out_feat, target_feat in zip(output_features, target_features):
            vgg_loss += F.mse_loss(out_feat, target_feat)
            
        # 扰动损失：对p个扰动输入计算损失
        perturbation_loss = 0
        for i in range(min(self.p, inputs.size(0))):
            input_features = self.get_features(inputs[i])
            for out_feat, input_feat in zip(output_features, input_features):
                perturbation_loss += F.mse_loss(out_feat, input_feat)
        
        # 总损失
        total_loss = self.alpha * (main_loss + vgg_loss) + (1 - self.alpha) * perturbation_loss
        
        return total_loss