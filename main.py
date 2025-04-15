#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import ctypes
import logging
import argparse
import math
from torch.utils.tensorboard import SummaryWriter
import traceback

# Windows内存优化设置
if os.name == 'nt':
    # 设置进程的工作集大小
    try:
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, 1<<30, 1<<30)  # 使用更小的值：1GB
    except Exception as e:
        logging.warning(f"设置进程工作集大小失败: {str(e)}")
    
    # 禁用内存压缩
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # 延迟加载CUDA模块

import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.cuda.amp import autocast, GradScaler
from Unetmodel import Unet
from setdata import LiverDataset, MmapLiverDataset
from customLoss import CustomLoss
import configparser
from datetime import datetime
import torch.cuda.amp as amp
import gc
import numpy as np
import random
import torch.nn.functional as F
from contextlib import nullcontext
import time
from colorama import init, Fore, Style

# 初始化colorama
init()

# 创建自定义的日志格式化器
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'INFO': Fore.GREEN,
        'DEBUG': Fore.WHITE
    }

    def format(self, record):
        # 为不同类型的消息添加颜色
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            # 特殊处理某些类型的消息
            if "GPU内存使用情况" in record.getMessage():
                color = Fore.CYAN
            elif "Epoch" in record.getMessage():
                if "验证损失" in record.getMessage():
                    color = Fore.MAGENTA
                else:
                    color = Fore.GREEN
            
            record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 设置确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 确保CUDA上的随机数生成器被正确初始化
        torch.cuda.manual_seed_all(seed)
        current_device = torch.cuda.current_device()
        torch.cuda.set_rng_state(torch.cuda.get_rng_state(current_device))

# 设置CUDA内存分配器
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 设置随机种子
set_seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置CUDA内存管理
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清空CUDA缓存

    torch.backends.cudnn.benchmark = False  # 使用cudnn自动寻找最适合当前配置的高效算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    
    # 限制内存使用
    torch.cuda.set_per_process_memory_fraction(0.7)  # 降低到70%
    
    # 主动垃圾回收
    gc.collect()
    torch.cuda.empty_cache()

# 设置日志
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# 数据预处理
x_transforms = None  # EXR图像的四个通道已经在read_exr中归一化
y_transforms = None  # GT已经在Dataset中被转换为tensor并归一化

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path, scheduler):
    """训练模型的主循环"""
    import gc
    import torch.cuda.amp as amp
    import logging
    from datetime import datetime
    
    # 检查GPU显存
    batch_size = train_loader.batch_size if train_loader else 1
    image_size = (4, 1024, 2048)  # 调整为实际输入尺寸2048×1024
    if not check_gpu_memory(model, batch_size, image_size, 
                          optimizer_type=optimizer.__class__.__name__.lower()):
        logging.error("GPU显存不足，无法开始训练")
        return
    
    # 启用梯度异常检测
    torch.autograd.set_detect_anomaly(True)
    
    
    # 创建TensorBoard writer
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', current_time)
    writer = SummaryWriter(log_dir)
    
    logging.basicConfig(level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler = logging.FileHandler('training.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    def log_gpu_memory():
        if device == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB
            logging.info(f'GPU内存使用情况 - 已分配: {allocated:.2f}MB, 已预留: {reserved:.2f}MB')
            writer.add_scalar('System/GPU_Memory_Allocated_MB', allocated, global_step)
            writer.add_scalar('System/GPU_Memory_Reserved_MB', reserved, global_step)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best_loss = float('inf')  # 使用训练损失或验证损失的最佳值
    global_step = 0
    
    # 定义梯度监控钩子 - 修改为不改变梯度值，只监控并记录
    def grad_hook(module, grad_input, grad_output):
        """实时梯度监控钩子 - 只监控不修改梯度"""
        try:
            for i, grad in enumerate(grad_input):
                if grad is None:
                    continue
                    
                # 检查并记录梯度统计信息
                if torch.isnan(grad).any():
                    logging.error(f"在模块 {module.__class__.__name__} 中检测到NaN梯度")
                elif torch.isinf(grad).any():
                    logging.error(f"在模块 {module.__class__.__name__} 中检测到Inf梯度")
                
                # 记录梯度范数信息
                with torch.no_grad():
                    try:
                        grad_norm = torch.norm(grad)
                        if grad_norm > 1e3:
                            logging.warning(f"模块 {module.__class__.__name__} 梯度范数过大: {grad_norm:.2f}")
                    except:
                        pass
        except Exception as e:
            logging.error(f"梯度监控钩子错误: {str(e)}")
            
        # 返回原始梯度，不做修改
        return grad_input

    # 为所有层注册钩子 - 扩大注册范围
    hooks = []
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.ModuleList) and not isinstance(layer, nn.Sequential) and not isinstance(layer, Unet):
            hook = layer.register_full_backward_hook(grad_hook)
            hooks.append(hook)
            
    # 如果使用增强损失函数，也为其注册钩子
    if hasattr(criterion, 'perturbation_loss'):
        for name, module in criterion.named_modules():
            if hasattr(module, 'parameters') and not isinstance(module, nn.ModuleList) and not isinstance(module, nn.Sequential):
                hook = module.register_full_backward_hook(grad_hook)
                hooks.append(hook)
        logging.info(f"已为损失函数注册梯度监控钩子")

    logging.info(f"总共为 {len(hooks)} 个模块注册了梯度监控钩子")
    
    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_l1_loss = 0.0
            batch_count = 0
            oom_recovery_attempts = 0  # 添加OOM恢复尝试计数器
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, 当前学习率: {current_lr}")
            
            # 训练阶段
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                try:
                    # 检查输入数据的有效性
                    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                        logging.warning(f"批次 {batch_idx} 的输入数据包含NaN或Inf，跳过")
                        continue
                    
                    if torch.isnan(labels).any() or torch.isinf(labels).any():
                        logging.warning(f"批次 {batch_idx} 的标签数据包含NaN或Inf，跳过")
                        continue
                    
                    # 记录输入数据的统计信息
                    if batch_idx % 10 == 0:
                        logging.info(f"输入数据统计 - 均值: {inputs.mean().item():.6f}, "
                                   f"标准差: {inputs.std().item():.6f}, "
                                   f"最大值: {inputs.max().item():.6f}, "
                                   f"最小值: {inputs.min().item():.6f}")
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    with torch.amp.autocast(device_type=device.type, 
                                     dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, 
                                     enabled=True):
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        
                        outputs = model(inputs)
                        
                        if hasattr(criterion, 'perturbation_loss'):
                            # 使用的是EnhancedCustomLoss
                            loss, loss_components = criterion(model, outputs, labels, inputs)
                            
                            l1_loss = loss_components['l1_loss']
                            vgg_loss = loss_components['vgg_loss']
                            perturbation_loss = loss_components['perturbation_loss']
                        else:
                            # 使用的是CustomLoss
                            loss = criterion(outputs, labels, inputs)
                            
                            l1_loss = criterion.l1(outputs, labels)
                            vgg_loss = (loss - criterion.alpha * l1_loss) / (1 - criterion.alpha)
                            perturbation_loss = torch.tensor(0.0, device=device)
                    
                    # 使用scaler来缩放损失和反向传播
                    scaler.scale(loss).backward(retain_graph=False)
                    
                    # 梯度处理和裁剪流程
                    try:
                        # 在优化器步骤前取消梯度缩放 - 确保只调用一次unscale_
                        # 检查梯度
                        valid_gradients = True
                        
                        # 先检查梯度是否包含NaN或Inf，不需要unscale
                        scaled_grads_have_inf_or_nan = False
                        has_severe_gradients = False
                        has_fixable_gradients = False
                        
                        # 第一阶段：检测是否存在无效梯度
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                nan_mask = torch.isnan(param.grad)
                                inf_mask = torch.isinf(param.grad)
                                
                                if nan_mask.any() or inf_mask.any():
                                    scaled_grads_have_inf_or_nan = True
                                    # 记录无效梯度比例
                                    invalid_ratio = (nan_mask.sum() + inf_mask.sum()).item() / param.grad.numel()
                                    
                                    # 如果无效值比例超过20%，认为这是严重问题
                                    if invalid_ratio > 0.2:
                                        has_severe_gradients = True
                                        logging.error(f"参数 {name} 中存在大量NaN或Inf梯度（{invalid_ratio*100:.1f}%），已跳过本批次更新")
                                        break
                                    else:
                                        has_fixable_gradients = True
                                        logging.warning(f"参数 {name} 中存在少量NaN或Inf梯度（{invalid_ratio*100:.1f}%），尝试修复")
                        
                        # 第二阶段：对于严重问题，直接跳过批次
                        if has_severe_gradients:
                            optimizer.zero_grad(set_to_none=True)
                            continue
                        
                        # 第三阶段：修复处理可修复的梯度问题
                        if has_fixable_gradients:
                            for name, param in model.named_parameters():
                                if param.grad is not None:
                                    grad_data = param.grad.data
                                    
                                    # 获取当前参数的有效梯度统计信息
                                    valid_mask = ~(torch.isnan(grad_data) | torch.isinf(grad_data))
                                    if valid_mask.sum() > 0:  # 确保有有效梯度
                                        valid_grads = grad_data[valid_mask]
                                        valid_mean = valid_grads.mean().item()
                                        valid_std = valid_grads.std().item() if valid_grads.numel() > 1 else 0.01
                                        
                                        # 将NaN替换为当前参数有效梯度的均值
                                        nan_mask = torch.isnan(grad_data)
                                        if nan_mask.any():
                                            # 加入少量噪声避免完全相同的值
                                            noise = torch.randn_like(grad_data[nan_mask]) * valid_std * 0.1
                                            grad_data[nan_mask] = valid_mean + noise
                                            
                                        # 将Inf替换为当前参数有效梯度的最大值的有符号倍数
                                        inf_mask = torch.isinf(grad_data)
                                        if inf_mask.any():
                                            max_valid = valid_grads.abs().max().item()
                                            sign_tensor = torch.sign(grad_data[inf_mask])
                                            # 使用有效梯度最大值的10倍(有符号)
                                            replacement = sign_tensor * max_valid * 10.0
                                            grad_data[inf_mask] = replacement
                                    else:
                                        # 如果没有有效梯度，则将所有梯度设为0
                                        grad_data.zero_()
                                        
                                    # 统计修复后的梯度分布
                                    if torch.isnan(grad_data).any() or torch.isinf(grad_data).any():
                                        logging.error(f"参数 {name} 的梯度修复失败，设置为零")
                                        grad_data.zero_()
                        
                        # 如果训练进行到后期（超过一半epoch），则进一步降低梯度裁剪阈值，防止后期不稳定
                        current_epoch_ratio = epoch / num_epochs
                        max_norm = 1.0 if current_epoch_ratio < 0.5 else max(0.1, 1.0 - current_epoch_ratio)
                        
                        # 预先对梯度进行裁剪（在取消缩放前），这有助于防止取消缩放后的梯度爆炸
                        scale = scaler.get_scale()
                        if math.isfinite(scale):  # 使用math.isfinite而不是torch.isfinite
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data.mul_(torch.clamp(torch.tensor(1.0 / max(1.0, torch.norm(param.grad.data) / (1000.0 * scale))), max=1.0))
                        
                        # 只有在梯度有效的情况下才unscale
                        scaler.unscale_(optimizer)
                        
                        # unscale后再次检查梯度
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                # 检查并处理非法梯度
                                if torch.isnan(param.grad).any():
                                    logging.error(f"参数 {name} 中存在NaN梯度，已跳过本批次更新")
                                    valid_gradients = False
                                    break
                                elif torch.isinf(param.grad).any():
                                    logging.error(f"参数 {name} 中存在Inf梯度，已跳过本批次更新")
                                    valid_gradients = False
                                    break
                                    
                                # 检查梯度范数
                                grad_norm = torch.norm(param.grad)
                                # 降低警告阈值，让更多问题提前暴露
                                if grad_norm > 1e3:  # 从1e4降低到1e3
                                    # 如果梯度范数非常大，直接跳过本批次
                                    if grad_norm > 1e5:
                                        logging.error(f"参数 {name} 梯度范数极大: {grad_norm:.2f}，跳过本批次")
                                        valid_gradients = False
                                        break
                                    else:
                                        logging.warning(f"参数 {name} 梯度范数过大: {grad_norm:.2f}")
                                        
                                        # 对梯度进行额外的缩放，防止极大值影响训练
                                        scale_factor = min(1.0, 1e3 / grad_norm)
                                        param.grad.data.mul_(scale_factor)
                        
                        if not valid_gradients:
                            # 放弃本批次更新
                            optimizer.zero_grad(set_to_none=True)
                            continue
                        
                        # 梯度裁剪 - 使用动态max_norm值
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                        
                        # 额外检查裁剪后的梯度是否合理
                        max_grad_norm = 0.0
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                grad_norm = torch.norm(param.grad).item()
                                max_grad_norm = max(max_grad_norm, grad_norm)
                        
                        # 如果最大梯度仍然过大，跳过本次更新
                        if max_grad_norm > 10.0:
                            logging.warning(f"裁剪后最大梯度范数仍然过大: {max_grad_norm:.2f}，跳过本批次")
                            optimizer.zero_grad(set_to_none=True)
                            continue
                            
                        # 执行优化器步骤
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    
                    except Exception as e:
                        logging.error(f"梯度处理出错: {str(e)}")
                        logging.error(traceback.format_exc())
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    running_loss += loss.item()
                    running_l1_loss += l1_loss.item()
                    batch_count += 1
                    
                    writer.add_scalar('Loss/Train/Total', loss.item(), global_step)
                    writer.add_scalar('Loss/Train/L1', l1_loss.item(), global_step)
                    writer.add_scalar('Loss/Train/VGG', vgg_loss.item(), global_step)
                    
                    # 记录扰动损失
                    if hasattr(criterion, 'perturbation_loss'):
                        writer.add_scalar('Loss/Train/Perturbation', perturbation_loss.item(), global_step)
                    
                    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                    
                    if global_step % 100 == 0:
                        writer.add_images('Images/Input', inputs[0:1, :3], global_step, dataformats='NCHW')
                        writer.add_images('Images/Input_Alpha', inputs[0:1, 3:], global_step, dataformats='NCHW')
                        
                        outputs_vis = outputs[0:1]
                        labels_vis = labels[0:1]
                        
                        # 保存时应该统一处理成与infer.py相同的格式
                        writer.add_images('Images/Prediction', outputs_vis, global_step, dataformats='NCHW')
                        writer.add_images('Images/Ground_Truth', labels_vis, global_step, dataformats='NCHW')
                        
                        # 同时保存一份应用缩放后的图像与推理脚本一致
                        outputs_vis_uint8 = (outputs_vis.detach().cpu() * 255).to(torch.uint8)
                        labels_vis_uint8 = (labels_vis.detach().cpu() * 255).to(torch.uint8)
                        writer.add_images('Images/Prediction_Uint8', outputs_vis_uint8, global_step, dataformats='NCHW')
                        writer.add_images('Images/Ground_Truth_Uint8', labels_vis_uint8, global_step, dataformats='NCHW')
                        
                        diff = torch.abs(outputs_vis - labels_vis)
                        writer.add_images('Images/Prediction_Diff', diff, global_step, dataformats='NCHW')
                        
                        logging.info(f"图像统计 - GT: min={labels_vis.min():.3f}, max={labels_vis.max():.3f}, mean={labels_vis.mean():.3f}")
                        logging.info(f"图像统计 - Pred: min={outputs_vis.min():.3f}, max={outputs_vis.max():.3f}, mean={outputs_vis.mean():.3f}")
                    
                    global_step += 1
                    
                    if batch_idx % 10 == 0:
                        avg_loss = running_loss / batch_count if batch_count > 0 else 0
                        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {avg_loss:.4f}')
                        
                    del inputs, labels, outputs, loss
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        oom_recovery_attempts += 1
                        if oom_recovery_attempts > 2:
                            logging.error("GPU内存不足且恢复尝试次数超过2次，停止训练")
                            logging.error(f"错误详情: {str(e)}")
                            # 清理GPU内存
                            if device == 'cuda':
                                torch.cuda.empty_cache()
                            # 关闭writer
                            writer.close()
                            # 立即退出程序
                            import sys
                            sys.exit(1)
                        
                        logging.warning(f"GPU内存不足，第{oom_recovery_attempts}次尝试恢复...")
                        for p in model.parameters():
                            if p.grad is not None:
                                del p.grad
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise e
                except Exception as e:
                    logging.error(f"处理批次 {batch_idx} 时出错: {str(e)}")
                    logging.error("错误详情: ", exc_info=True)
                    continue
            
            epoch_loss = running_loss / batch_count if batch_count > 0 else float('inf')
            epoch_l1_loss = running_l1_loss / batch_count if batch_count > 0 else float('inf')
            
            writer.add_scalar('Loss/Train/Epoch_Avg_Total', epoch_loss, epoch)
            writer.add_scalar('Loss/Train/Epoch_Avg_L1', epoch_l1_loss, epoch)
            
            logging.info(f'Epoch [{epoch+1}/{num_epochs}] 平均训练损失: {epoch_loss:.4f}')
            
            # 验证阶段
            if val_loader is not None:
                avg_val_loss = validate_direct(model, val_loader, criterion, device, 
                                              global_step=global_step, 
                                              writer=writer)
                
                writer.add_scalar('Loss/Val/Total', avg_val_loss, epoch)
                
                # 更新学习率
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()
                
                # 记录当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning_Rate', current_lr, epoch)
                
                logging.info(f'Epoch [{epoch+1}/{num_epochs}] 验证损失: {avg_val_loss:.4f}, 学习率: {current_lr:.6f}')
                
                # 保存最佳模型
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                        }, save_path)
                        logging.info(f'保存最佳模型，验证损失: {best_loss:.4f}')
                    except Exception as e:
                        logging.error(f"保存模型时出错: {str(e)}")
            else:
                # 无验证集时，使用训练损失来保存最佳模型
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                        }, save_path)
                        logging.info(f'保存最佳模型，训练损失: {best_loss:.4f}')
                    except Exception as e:
                        logging.error(f"保存模型时出错: {str(e)}")
                
                # 更新学习率（如果不是ReduceLROnPlateau）
                if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('Learning_Rate', current_lr, epoch)
                    logging.info(f'Epoch [{epoch+1}/{num_epochs}] 学习率: {current_lr:.6f}')
            
            # 主动进行垃圾回收
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        logging.info("训练被用户中断")
    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}")
        raise
    
    return model

def validate_direct(model, val_loader, criterion, device, global_step=0, writer=None):
    """直接进行验证，不使用分块处理
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        global_step: 当前全局步数
        writer: TensorBoard SummaryWriter对象，如果为None则创建新的
    """
    model.eval()
    val_loss = 0.0
    val_l1_loss = 0.0
    val_perturb_loss = 0.0
    val_batch_count = 0
    
    has_perturbation = hasattr(criterion, 'perturbation_loss')
    
    try:
        with torch.inference_mode():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type=device.type, 
                                     dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, 
                                     enabled=True):
                    outputs = model(val_inputs)
                    
                    if has_perturbation:
                        # 使用的是EnhancedCustomLoss
                        loss, loss_components = criterion(model, outputs, val_labels, val_inputs)
                        l1_loss = loss_components['l1_loss']
                        perturb_loss = loss_components['perturbation_loss']
                    else:
                        # 使用的是CustomLoss
                        loss = criterion(outputs, val_labels, val_inputs)
                        l1_loss = criterion.l1(outputs, val_labels)
                        perturb_loss = torch.tensor(0.0, device=device)
                
                val_loss += loss.item()
                val_l1_loss += l1_loss.item()
                if has_perturbation:
                    val_perturb_loss += perturb_loss.item()
                val_batch_count += 1
                
                # 及时清理内存
                del val_inputs, val_labels, outputs, loss, l1_loss
                if has_perturbation:
                    del perturb_loss, loss_components
                torch.cuda.empty_cache()
                
    except Exception as e:
        logging.error(f"验证时出错: {str(e)}")
        logging.error(traceback.format_exc())
        # 继续执行，避免单个错误导致整个训练中断
    
    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
    avg_val_l1_loss = val_l1_loss / val_batch_count if val_batch_count > 0 else float('inf')
    
    # 记录到TensorBoard中的验证损失细节
    if writer is None:
        writer = SummaryWriter(os.path.join('runs', 'validation'))
        needs_close = True
    else:
        needs_close = False
    
    writer.add_scalar('Loss/Val/Total_Direct', avg_val_loss, global_step)
    writer.add_scalar('Loss/Val/L1_Direct', avg_val_l1_loss, global_step)
    
    if has_perturbation:
        avg_val_perturb_loss = val_perturb_loss / val_batch_count if val_batch_count > 0 else 0.0
        writer.add_scalar('Loss/Val/Perturbation_Direct', avg_val_perturb_loss, global_step)
        logging.info(f'直接验证损失: 总计={avg_val_loss:.4f}, L1={avg_val_l1_loss:.4f}, 扰动={avg_val_perturb_loss:.4f}')
    else:
        logging.info(f'直接验证损失: 总计={avg_val_loss:.4f}, L1={avg_val_l1_loss:.4f}')
    
    if needs_close:
        writer.close()
    
    return avg_val_loss

def estimate_memory_usage(model, image_size, batch_size, is_training=True, optimizer_type='adam'):
    """精确显存估算函数
    
    Args:
        model: 模型实例
        image_size: (channels, height, width)
        batch_size: 批次大小
        is_training: 是否训练模式
        optimizer_type: 优化器类型（影响优化器状态）
    """
    try:
        # 基础参数计算
        channels, height, width = image_size
        input_size = batch_size * channels * height * width * 4  # float32
        
        # 模型参数内存（训练模式包含梯度）
        param_mem = sum(p.numel() * 4 for p in model.parameters())  # 参数本身
        if is_training:
            param_mem *= 2  # 梯度内存
        
        # 优化器状态内存
        if is_training:
            if optimizer_type.lower() == 'adam':
                # Adam需要保存m和v，每个参数额外8字节
                optimizer_mem = sum(p.numel() * 8 for p in model.parameters())
            else:
                # SGD只需要动量，每个参数4字节
                optimizer_mem = sum(p.numel() * 4 for p in model.parameters())
        else:
            optimizer_mem = 0
        
        # 激活值估算（U-Net特殊处理）
        # 计算公式：输入大小 * 网络深度系数 * 跳跃连接系数
        activation_factor = 18  # U-Net的激活量约为输入的18倍
        activation_mem = input_size * activation_factor
        
        # 框架基础开销（CUDA上下文+内部缓存）
        framework_overhead = 512 * 1024 * 1024  # 512MB
        
        # 总内存计算
        total_mem = (
            input_size + 
            param_mem + 
            optimizer_mem + 
            activation_mem + 
            framework_overhead
        )
        
        return total_mem / (1024 ** 2)  # 转换为MB
        
    except Exception as e:
        logging.error(f"显存估算出错: {str(e)}")
        return float('inf')

def check_gpu_memory(model, batch_size, image_size, optimizer_type='adam'):
    """检查GPU显存是否足够运行训练"""
    if not torch.cuda.is_available():
        return True
    
    try:
        # 获取当前GPU可用显存
        gpu = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(gpu)
        total_memory = gpu_properties.total_memory / (1024 * 1024)  # 转换为MB
        allocated_memory = torch.cuda.memory_allocated(gpu) / (1024 * 1024)
        cached_memory = torch.cuda.memory_reserved(gpu) / (1024 * 1024)
        free_memory = total_memory - allocated_memory - cached_memory
        
        # 估算训练所需显存
        estimated_memory = estimate_memory_usage(
            model=model,
            image_size=image_size,
            batch_size=batch_size,
            is_training=True,
            optimizer_type=optimizer_type
        )
        
        # 打印显存信息
        logging.info(f"GPU显存状态:")
        logging.info(f"  总显存: {total_memory:.2f}MB")
        logging.info(f"  已分配: {allocated_memory:.2f}MB")
        logging.info(f"  缓存: {cached_memory:.2f}MB")
        logging.info(f"  可用: {free_memory:.2f}MB")
        logging.info(f"  预估训练需要: {estimated_memory:.2f}MB")
        
        # 预留20%的缓冲区
        buffer_ratio = 1.2
        required_memory = estimated_memory * buffer_ratio
        
        if required_memory > free_memory:
            logging.warning(f"警告：预估所需显存({required_memory:.2f}MB) 超过可用显存({free_memory:.2f}MB)")
            logging.warning("建议减小batch_size或使用梯度累积")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"检查GPU显存时出错: {str(e)}")
        return False

def find_optimal_batch_size(model, image_size, available_memory, start_batch_size=1):
    """
    根据可用显存自动寻找最优批量大小
    """
    try:
        current_batch_size = start_batch_size
        while True:
            estimated_memory = estimate_memory_usage(model, image_size, current_batch_size)
            if estimated_memory > available_memory * 0.85:  # 保留15%的显存余量
                return max(1, current_batch_size - 1)
            current_batch_size *= 2
            
    except Exception as e:
        logging.error(f"批量大小优化出错: {str(e)}")
        return start_batch_size

def load_dataset(processed_dir, batch_size, num_workers=0, apply_normalization=True):
    """
    加载预处理后的数据集
    
    Args:
        processed_dir: 预处理数据目录 (例如: ./data/processed)
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        apply_normalization: 是否应用数据标准化
    """
    # 统计文件路径
    stats_path = os.path.join(processed_dir, 'train_stats.npy') 
    
    # 检查统计文件是否存在，如果需要标准化但文件不存在，则报错退出
    if apply_normalization and not os.path.exists(stats_path):
        logging.error(f"错误：需要进行标准化，但未找到数据集统计文件: {stats_path}")
        logging.error("请先运行 calculate_dataset_stats.py 脚本生成该文件。")
        raise FileNotFoundError(f"未找到数据集统计文件: {stats_path}")
    elif not apply_normalization:
         logging.warning("注意：未启用数据标准化 (apply_normalization=False)。")

    # processed_dir 同时作为数据目录和统计目录传递给 Dataset
    data_dir = processed_dir
    stats_dir = processed_dir 
    
    # 创建训练集数据集对象
    try:
        logging.info(f"尝试加载训练集...")
        train_dataset = MmapLiverDataset(
            data_dir=data_dir,
            split='train', # <--- 指定加载训练集
            stats_dir=stats_dir, # <--- 指定统计文件目录
            transform=None,
            target_transform=None,
            apply_normalization=apply_normalization
        )
        logging.info(f"训练集加载成功，样本数: {len(train_dataset)}")
        
        # 创建验证集数据集对象
        logging.info(f"尝试加载验证集...")
        val_dataset = MmapLiverDataset(
            data_dir=data_dir,
            split='val', # <--- 指定加载验证集
            stats_dir=stats_dir, # <--- 指定统计文件目录
            transform=None,
            target_transform=None,
            apply_normalization=apply_normalization # 使用与训练集相同的标准化
        )
        logging.info(f"验证集加载成功，样本数: {len(val_dataset)}")
        
    except FileNotFoundError as e:
        logging.error(f"加载数据集文件失败: {e}")
        logging.error(f"请确保以下文件存在于 '{processed_dir}' 目录中:")
        logging.error(f"  - train_inputs.npy")
        logging.error(f"  - train_labels.npy")
        logging.error(f"  - val_inputs.npy")
        logging.error(f"  - val_labels.npy")
        if apply_normalization:
             logging.error(f"  - train_stats.npy")
        raise e
    except Exception as e:
        logging.error(f"使用 MmapLiverDataset 加载数据集时发生未知错误: {str(e)}")
        raise e

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False, # 如果需要打乱，设为 True
        num_workers=num_workers,
        pin_memory=False, 
        prefetch_factor=None, 
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=False, 
        prefetch_factor=None, 
        persistent_workers=False
    )
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='训练模型')
    parser.add_argument('--loss_type', type=str, default=None, help='损失函数类型')
    parser.add_argument('--perturb_weight', type=float, default=None, help='扰动损失权重')
    args = parser.parse_args()

    # 读取配置文件
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    
    # 使用命令行参数覆盖配置文件设置
    loss_type = args.loss_type if args.loss_type is not None else config.get('base', 'loss_type', fallback='standard')
    perturb_weight = args.perturb_weight if args.perturb_weight is not None else float(config.get('base', 'perturb_weight', fallback='0.1'))
    
    # 读取dropout率和优化器类型
    dropout_rate = float(config.get('base', 'dropout_rate', fallback='0.2'))
    optimizer_type = config.get('base', 'optimizer_type', fallback='adam')
    
    # 设置日志
    log_dir = config.get('base', 'log_dir')
    setup_logger(log_dir)
    
    # 初始化模型
    input_channels = int(config.get('base', 'input_channels'))
    output_channels = int(config.get('base', 'output_channels'))
    model = Unet(input_channels, output_channels, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # 获取图像尺寸
    image_height = int(config.get('base', 'image_height'))
    image_width = int(config.get('base', 'image_width'))
    image_size = (1, input_channels, image_height, image_width)
    
    # 估算显存使用量并优化批量大小
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # 转换为MB
        initial_batch_size = int(config.get('base', 'batch_size'))
        
        # 估算当前批量大小的显存使用量
        estimated_memory = estimate_memory_usage(model, image_size, initial_batch_size)
        logging.info(f"预计显存使用量: {estimated_memory:.2f}MB (批量大小={initial_batch_size})")
        
        # 寻找最优批量大小
        optimal_batch_size = find_optimal_batch_size(model, image_size, available_memory, initial_batch_size)
        
        if optimal_batch_size != initial_batch_size:
            logging.warning(f"将批量大小从{initial_batch_size}调整为{optimal_batch_size}")
            if optimal_batch_size < initial_batch_size:
                logging.warning("当前批量大小可能导致显存溢出")
        
        batch_size = initial_batch_size
    else:
        batch_size = int(config.get('base', 'batch_size'))
    
    # 完全禁用多进程加载
    num_workers = 0
    
    processed_data_path = config['base']['processed_data_dir']

    # 加载数据集 (假设标准化默认启用)
    train_loader, val_loader = load_dataset(
        processed_dir=processed_data_path, 
        batch_size=batch_size,
        num_workers=num_workers,
        apply_normalization=True # 明确启用标准化
    )
    
    # 定义损失函数和优化器
    if loss_type == 'perturb':
        from pert_loss import EnhancedCustomLoss
        criterion = EnhancedCustomLoss(device, alpha=float(config.get('base', 'alpha', fallback='0.9')), 
                                      perturb_weight=perturb_weight)
        logging.info(f"使用增强版损失函数(L1 + VGG + 扰动损失)，扰动权重: {perturb_weight}")
    else:
        from customLoss import CustomLoss
        criterion = CustomLoss(device, alpha=float(config.get('base', 'alpha', fallback='0.9')))
        logging.info("使用标准损失函数(L1 + VGG)")
    
    # 添加学习率预热和更严格的学习率衰减策略
    warmup_epochs = int(config.get('base', 'warmup_epochs', fallback=5))
    num_epochs = int(config.get('base', 'num_epochs'))
    initial_learning_rate = float(config.get('base', 'learning_rate'))
    logging.info(f"初始化优化器，学习率: {initial_learning_rate}")
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=1e-4)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=1e-3)
    else:
        optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=0.9, weight_decay=1e-4)
    
    def get_lr_lambda(epoch):
        """学习率调度器函数"""
        if epoch < warmup_epochs:
            # 预热期学习率线性增加
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # 预热后使用余弦退火，最小学习率为初始学习率的1%
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
            return max(0.01, decay_factor)  # 确保学习率不低于初始值的1%
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    logging.info(f"已创建学习率调度器: 预热轮数={warmup_epochs}, 总轮数={num_epochs}")
    
    logging.info(f"初始学习率设置为: {initial_learning_rate}")
    
    # 开始训练
    logging.info(f"Starting training on device: {device}")
    save_path = os.path.join(config.get('base', 'save_dir'), 'best_model.pth')

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path, scheduler)
    logging.info("Training completed!")

if __name__ == "__main__":
    main()