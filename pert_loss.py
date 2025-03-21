import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random

class PerturbationLoss(nn.Module):
    """
    扰动损失实现 - 基于论文《Neural Shadow Mapping》(Datta et al., 2022)
    该损失函数通过对输入进行小扰动并比较扰动前后的输出差异来提高模型的时间稳定性
    """
    def __init__(self, perturbation_count=3, alpha=0.9):
        """
        初始化扰动损失
        
        Args:
            perturbation_count: 生成的扰动输入数量
            alpha: 在感知损失中权衡L1损失和VGG损失的系数
        """
        super().__init__()
        self.perturbation_count = perturbation_count
        self.alpha = alpha
        self.loss_fn = nn.L1Loss()
        logging.info(f"初始化扰动损失，扰动数量: {perturbation_count}")
        
    def perturb_input(self, x, std_factor=0.01):
        """
        对输入进行小扰动
        
        Args:
            x: 输入张量 [B, C, H, W]
            std_factor: 扰动标准差因子
            
        Returns:
            perturbed_inputs: 扰动后的输入列表
        """
        perturbed_inputs = []
        batch_size, channels, height, width = x.shape
        
        # 为每个通道计算标准差
        channel_stds = []
        for c in range(channels):
            channel_data = x[:, c, :, :]
            std = torch.std(channel_data).item()
            channel_stds.append(std)
            
        logging.debug(f"输入通道标准差: {channel_stds}")
        
        # 生成多个扰动输入
        for i in range(self.perturbation_count):
            # 为每个通道创建扰动
            perturbed_tensor = x.clone()
            for c in range(channels):
                noise = torch.randn_like(x[:, c:c+1, :, :]) * channel_stds[c] * std_factor
                perturbed_tensor[:, c:c+1, :, :] += noise
            
            perturbed_inputs.append(perturbed_tensor)
            
        return perturbed_inputs
            
    def forward(self, model, original_input, original_output):
        """
        计算扰动损失
        
        Args:
            model: 要评估的模型
            original_input: 原始输入 [B, C, H, W]
            original_output: 模型对原始输入的输出 [B, 1, H, W]
            
        Returns:
            loss: 扰动损失值
        """
        # 创建输入扰动
        perturbed_inputs = self.perturb_input(original_input)
        
        # 计算每个扰动输入的输出
        perturbed_outputs = []
        with torch.no_grad():  # 不需要为扰动输入计算梯度
            for p_input in perturbed_inputs:
                p_output = model(p_input)
                perturbed_outputs.append(p_output)
                
        # 计算原始输出与扰动输出之间的差异
        total_perturbation_loss = 0
        for p_output in perturbed_outputs:
            loss = self.loss_fn(original_output, p_output)
            total_perturbation_loss += loss
            
        # 返回平均扰动损失
        return total_perturbation_loss / len(perturbed_outputs)
        
class EnhancedCustomLoss(nn.Module):
    """
    增强版自定义损失函数，结合L1损失、VGG损失和扰动损失
    """
    def __init__(self, device, alpha=0.9, perturb_weight=0.5):
        """
        初始化增强版自定义损失
        
        Args:
            device: 计算设备
            alpha: L1损失和VGG损失之间的权重
            perturb_weight: 扰动损失的权重
        """
        super().__init__()
        self.alpha = alpha
        self.perturb_weight = perturb_weight
        self.l1 = nn.L1Loss()
        
        # 导入需要的模块 - 确保不破坏循环导入
        from customLoss import VGGLoss
        self.vgg_loss = VGGLoss(device)
        
        self.perturbation_loss = PerturbationLoss()
        logging.info(f"初始化增强版损失函数: alpha={alpha}, perturb_weight={perturb_weight}")
        
    def forward(self, model, output, target, inputs):
        """
        计算增强版损失
        
        Args:
            model: 用于计算扰动损失的模型
            output: 模型输出 [B, 1, H, W]
            target: 目标阴影 [B, 1, H, W]
            inputs: 模型输入 [B, C, H, W]
            
        Returns:
            loss: 总损失
        """
        # 确保输出经过Sigmoid
        assert output.min() >= 0 and output.max() <= 1, "输出必须经过Sigmoid激活!"
        
        # 计算L1损失
        l1 = self.l1(output, target)
        
        # 计算VGG损失
        vgg = self.vgg_loss(output, target, inputs)
        
        # 计算基本损失（L1 + VGG）
        basic_loss = self.alpha * l1 + (1 - self.alpha) * vgg
        
        # 计算扰动损失
        if self.training and self.perturb_weight > 0:
            perturbation_loss = self.perturbation_loss(model, inputs, output)
            total_loss = basic_loss + self.perturb_weight * perturbation_loss
            
            logging.debug(f"损失详情: L1={l1.item():.6f}, VGG={vgg.item():.6f}, " 
                         f"扰动={perturbation_loss.item():.6f}, 总计={total_loss.item():.6f}")
            
            return total_loss
        else:
            logging.debug(f"损失详情: L1={l1.item():.6f}, VGG={vgg.item():.6f}, 总计={basic_loss.item():.6f}")
            return basic_loss

# 用于测量时间稳定性的函数
def measure_temporal_instability(frames, motion_vectors=None, alpha=5.0):
    """
    测量视频序列中的时间不稳定性
    
    Args:
        frames: 帧张量列表 [T, B, 1, H, W]
        motion_vectors: 可选，运动向量列表 [T-1, B, 2, H, W]
        alpha: 指数权重因子
    
    Returns:
        instability: 时间不稳定性指标
    """
    if len(frames) < 2:
        return torch.tensor(0.0)
    
    total_diff = 0
    for t in range(1, len(frames)):
        if motion_vectors is not None:
            # 基于运动向量进行调整的差异计算
            # 运动向量实现需要根据实际情况调整
            pass
        else:
            # 简单的帧间差异
            diff = torch.abs(frames[t] - frames[t-1])
        
        # 应用指数惩罚大的差异
        weighted_diff = torch.exp(alpha * diff) - 1
        total_diff += torch.mean(weighted_diff)
    
    return total_diff / (len(frames) - 1)
