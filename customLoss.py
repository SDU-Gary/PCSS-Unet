import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from pytorch_msssim import ssim

class MultiLayerVGGLoss(nn.Module):
    def __init__(self, device, feature_layers=(2, 7, 12, 21, 30), weights=(0.25, 0.25, 0.3, 0.1, 0.1)):
        """使用多个VGG特征层的损失函数
        
        Args:
            device: 计算设备
            feature_layers: 使用的VGG特征层索引
            weights: 各层特征的权重，长度需与feature_layers相同
        """
        super().__init__()
        assert len(feature_layers) == len(weights), "特征层和权重数量必须相同"
        
        # 加载预训练VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        
        # 为每个特征层创建子网络
        self.feature_extractors = nn.ModuleList()
        for layer_idx in feature_layers:
            # 截取从开始到指定层的子网络
            layers = nn.Sequential(*list(vgg.children())[:layer_idx+1])
            # 冻结参数
            for param in layers.parameters():
                param.requires_grad = False
            # 添加到模块列表
            self.feature_extractors.append(layers.to(device))
        
        # 保存权重 - 确保权重总和为1，并且整体缩小以降低损失比例
        weights_tensor = torch.tensor(weights)
        normalized_weights = weights_tensor / weights_tensor.sum()
        self.register_buffer('weights', normalized_weights.to(device))
        
        # 注册归一化参数 - 使用与ImageNet更接近的参数，但适用于灰度图像
        self.register_buffer('mean', torch.tensor([0.485]).view(1,1,1,1).to(device))  # 使用与RGB接近的均值
        self.register_buffer('std', torch.tensor([0.229]).view(1,1,1,1).to(device))   # 使用与RGB接近的标准差
    
    def forward(self, output, target):
        # 确保输入类型一致并裁剪到有效范围
        output = torch.clamp(output.to(torch.float32), 0.0, 1.0)
        target = torch.clamp(target.to(torch.float32), 0.0, 1.0)
        
        # 安全检查并修复NaN/Inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
            
        if torch.isnan(target).any() or torch.isinf(target).any():
            target = torch.nan_to_num(target, nan=0.5, posinf=1.0, neginf=0.0)
        
        # 单通道灰度图扩展为3通道以适应VGG
        output_3ch = output.repeat(1, 3, 1, 1)
        target_3ch = target.repeat(1, 3, 1, 1)
        
        # 归一化
        epsilon = 1e-8
        output_norm = (output_3ch - self.mean) / (self.std + epsilon)
        target_norm = (target_3ch - self.mean) / (self.std + epsilon)
        
        # 多层特征提取与损失计算 - 避免使用就地操作
        total_loss = 0.0
        
        for i, extractor in enumerate(self.feature_extractors):
            try:
                # 提取特征
                with torch.no_grad():  # 在特征提取过程中不计算梯度以节省内存
                    output_feat = extractor(output_norm)
                    target_feat = extractor(target_norm)
                    
                    # 检查特征尺度并归一化以减少层间差异
                    #output_feat = F.layer_norm(output_feat, output_feat.size()[1:])
                    #target_feat = F.layer_norm(target_feat, target_feat.size()[1:])
                
                # 再次检查并修复特征中的异常值
                output_feat = torch.nan_to_num(output_feat, nan=0.0, posinf=1.0, neginf=-1.0)
                target_feat = torch.nan_to_num(target_feat, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 计算MSE损失并加权
                layer_loss = F.l1_loss(output_feat, target_feat)
                total_loss = total_loss + self.weights[i] * layer_loss
            except Exception as e:
                print(f"VGG特征层{i}提取错误: {e}")
                # 出错时添加一个小的常数损失 - 避免就地操作
                total_loss = total_loss + 0.002
        
        # 将最终结果转换为需要梯度的张量
        return torch.tensor(total_loss, device=output.device, requires_grad=True)

class CustomLoss(nn.Module):
    def __init__(self, device, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.vgg_loss = MultiLayerVGGLoss(device)
        # self.ssim_loss = ssim
        
        # 添加高斯核用于高频信息提取
        kernel_size = 5
        sigma = 1.0
        self.device = device
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel(kernel_size, sigma))
        
    def _create_gaussian_kernel(self, kernel_size, sigma):
        """创建二维高斯核"""
        x = torch.arange(kernel_size) - kernel_size // 2
        x = x.expand(kernel_size, -1)
        y = x.transpose(0, 1)
        
        kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)
    
    def _extract_high_freq(self, img):
        """提取图像高频细节"""
        # 确保输入是单通道
        if img.shape[1] > 1:
            img = torch.mean(img, dim=1, keepdim=True)
            
        # 高斯模糊
        blurred = F.conv2d(img, self.gaussian_kernel, padding=2)
        
        # 高频 = 原图 - 模糊图
        high_freq = img - blurred
        return high_freq
        
    def forward(self, output, target, inputs):
        # 确保输出经过Sigmoid
        assert output.min() >= 0 and output.max() <= 1, "输出必须经过Sigmoid激活!"
        
        # 计算L1损失
        l1 = self.l1(output, target)
        
        # 计算VGG损失 - 直接使用输出和目标
        vgg = self.vgg_loss(output, target)
        
        # 计算高频细节损失
        output_high_freq = self._extract_high_freq(output)
        target_high_freq = self._extract_high_freq(target)
        high_freq_loss = F.l1_loss(output_high_freq, target_high_freq)
        
        # 软阴影区域增强
        penumbra_mask = (target > 0.1) & (target < 0.9)  # 半影区域
        penumbra_mask = penumbra_mask.float()
        
        # 对半影区域的L1损失加权
        penumbra_l1 = F.l1_loss(
            output * penumbra_mask, 
            target * penumbra_mask, 
            reduction='sum'
        ) / (penumbra_mask.sum() + 1e-8)
        
        # 在不修改主要权重的情况下，将细节损失和半影损失融入总损失
        # 使用较小的权重以避免破坏原始损失结构
        detail_weight = 0.1  # 小权重，保持总体训练稳定性
        
        # 组合损失，保持原有的alpha比例
        base_loss = self.alpha * l1 + (1 - self.alpha) * vgg
        # enhanced_loss = base_loss + detail_weight * (high_freq_loss + penumbra_l1)

        def compute_sobel(img):
            # 确保单通道灰度图
            if img.shape[1] > 1: img = torch.mean(img, dim=1, keepdim=True)
            # 定义 Sobel 算子 (或者从 buffer 读取)
            # 使用与图像相同的数据类型和设备
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
            # 应用卷积计算梯度
            grad_x = F.conv2d(img, sobel_x, padding=1)
            grad_y = F.conv2d(img, sobel_y, padding=1)
            # 计算梯度幅值
            magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6) # 加 epsilon 防 NaN
            return magnitude

        with torch.no_grad(): # GT 的梯度不需要计算图
            target_grad = compute_sobel(target)
        output_grad = compute_sobel(output)

        gradient_loss = F.l1_loss(output_grad, target_grad)

        # 调整梯度损失的权重
        gradient_weight = 0.1 # 可以尝试 0.05, 0.1, 0.2 等值
        enhanced_loss = base_loss + gradient_weight * gradient_loss

        # ssim_val = self.ssim_loss(output, target)
        # ssim_loss_term = 1.0 - ssim_val # SSIM 值越大越好，损失是 1 - SSIM

        # ssim_weight = 0.1 # 调整权重
        # final_loss = enhanced_loss + ssim_weight * ssim_loss_term
        
        return base_loss

class EnhancedCustomLoss(nn.Module):
    def __init__(self, device, alpha=0.9, beta=0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()
        self.vgg_loss = MultiLayerVGGLoss(device)
        
    def forward(self, model, output, target, inputs):
        # 计算基础损失
        l1_loss = self.l1(output, target)
        vgg_loss = self.vgg_loss(output, target)
        
        # 计算扰动损失（时间稳定性）
        perturbation_loss = self.compute_perturbation_loss(model, output, inputs)
        
        # 组合损失
        total_loss = self.alpha * l1_loss + (1 - self.alpha) * vgg_loss + self.beta * perturbation_loss
        
        # 返回总损失和各组件损失
        loss_components = {
            'l1_loss': l1_loss,
            'vgg_loss': vgg_loss,
            'perturbation_loss': perturbation_loss
        }
        
        return total_loss, loss_components
    
    def compute_perturbation_loss(self, model, output, inputs):
        """计算扰动损失，提高时间稳定性"""
        # 创建输入的轻微扰动版本
        epsilon = 0.01
        noise = torch.randn_like(inputs) * epsilon
        perturbed_inputs = inputs + noise
        
        # 确保数值稳定性
        perturbed_inputs = torch.clamp(perturbed_inputs, -10.0, 10.0)
        
        # 前向传播获取扰动输出
        with torch.no_grad():  # 避免计算二阶梯度
            perturbed_output = model(perturbed_inputs)
        
        # 计算原始输出和扰动输出之间的差异
        return F.mse_loss(output, perturbed_output)