#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import ctypes
import logging

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
from torch.utils.tensorboard import SummaryWriter
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
    image_size = (4, 512, 512)  # 根据实际输入调整
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
    
    # 定义梯度监控钩子
    def grad_hook(module, grad_input, grad_output):
        """实时梯度监控钩子"""
        def process_gradient(grad):
            if grad is None:
                return None
                
            # 创建梯度的深度副本
            processed_grad = grad.detach().clone()
            
            # 检查并记录梯度统计信息
            if torch.isnan(processed_grad).any() or torch.isinf(processed_grad).any():
                with torch.no_grad():
                    grad_stats = {
                        'mean': processed_grad.abs().mean().item() if not torch.isnan(processed_grad.abs().mean()) else 'NaN',
                        'max': processed_grad.abs().max().item() if not torch.isnan(processed_grad.abs().max()) else 'NaN',
                        'min': processed_grad.abs().min().item() if not torch.isnan(processed_grad.abs().min()) else 'NaN',
                        'std': processed_grad.std().item() if not torch.isnan(processed_grad.std()) else 'NaN'
                    }
                    logging.error(f"在模块 {module.__class__.__name__} 中检测到非法梯度值:")
                    logging.error(f"梯度统计: {grad_stats}")
            
            with torch.no_grad():
                # 处理NaN
                mask_nan = torch.isnan(processed_grad)
                if mask_nan.any():
                    processed_grad = torch.where(mask_nan, torch.zeros_like(processed_grad), processed_grad)
                    logging.warning(f"模块 {module.__class__.__name__} 中的NaN梯度已替换为0")
                
                # 处理Inf
                mask_inf = torch.isinf(processed_grad)
                if mask_inf.any():
                    max_val = 1e4
                    processed_grad = torch.clamp(
                        torch.where(mask_inf, torch.sign(processed_grad) * max_val, processed_grad),
                        -max_val, max_val
                    )
                    logging.warning(f"模块 {module.__class__.__name__} 中的Inf梯度已裁剪到[-{max_val}, {max_val}]范围内")
                
                # 检查梯度范数并缩放
                grad_norm = torch.norm(processed_grad)
                if grad_norm > 1e4:
                    scale_factor = 1e4 / grad_norm
                    processed_grad = processed_grad * scale_factor
                    logging.info(f"模块 {module.__class__.__name__} 的梯度已缩放，范数从 {grad_norm:.2f} 降至 1e4")
            
            return processed_grad

        try:
            # 处理所有输入梯度
            processed_grads = tuple(process_gradient(g) for g in grad_input)
            
            # 验证处理后的梯度
            for g in processed_grads:
                if g is not None and (torch.isnan(g).any() or torch.isinf(g).any()):
                    logging.error(f"模块 {module.__class__.__name__} 的梯度处理失败，仍存在非法值")
                    return grad_input  # 如果处理失败，返回原始梯度
            
            return processed_grads
            
        except Exception as e:
            logging.error(f"梯度钩子中发生错误: {str(e)}")
            return grad_input  # 发生错误时返回原始梯度

    # 为相关层注册钩子
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.InstanceNorm2d, nn.AvgPool2d)):
            hook = layer.register_full_backward_hook(grad_hook)
            hooks.append(hook)
            logging.info(f"已为层 {name} ({layer.__class__.__name__}) 注册梯度监控钩子")
    
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
                    
                    with torch.amp.autocast('cuda', 
                                        dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, 
                                        enabled=True):
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels, inputs)
                        
                        l1_loss = criterion.l1(outputs, labels)
                        vgg_loss = (loss - criterion.alpha * l1_loss) / (1 - criterion.alpha)
                    
                    # 使用scaler来缩放损失和反向传播
                    scaler.scale(loss).backward(retain_graph=False)
                    
                    try:
                        # 在scaler.step之前执行梯度检查
                        if scaler.is_enabled():
                            # 检查缩放后的梯度
                            unscaled_grads = []
                            for param in model.parameters():
                                if param.grad is not None:
                                    unscaled_grads.append(param.grad.detach().clone())
                            
                            # 统一缩放因子检查
                            inv_scale = 1. / scaler.get_scale()
                            max_unscaled_grad = max(torch.max(torch.abs(g * inv_scale)) for g in unscaled_grads)
                            if max_unscaled_grad > 1e4:
                                logging.warning(f"未缩放梯度过大: {max_unscaled_grad.item():.2f}")
                                scaler.update(0.5 * scaler.get_scale())  # 主动降低缩放因子
                                continue
                        
                        # 在更新权重之前取消缩放梯度
                        scaler.unscale_(optimizer)
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            max_norm=1.0,
                            norm_type=2.0,  # 使用L2范数
                            error_if_nonfinite=True
                        )
                        
                        # 执行优化器步骤
                        scaler.step(optimizer)
                        scaler.update()
                        
                    except RuntimeError as e:
                        if '非有限梯度值' in str(e):
                            logging.critical("实时梯度监控发现异常，执行紧急恢复")
                            # 1. 清除梯度
                            optimizer.zero_grad(set_to_none=True)
                            # 2. 重置缩放器
                            scaler.update(2.**10)  # 重置到较低缩放因子
                            # 3. 跳过当前批次
                            continue
                        else:
                            raise e
                    
                    running_loss += loss.item()
                    running_l1_loss += l1_loss.item()
                    batch_count += 1
                    
                    writer.add_scalar('Loss/Train/Total', loss.item(), global_step)
                    writer.add_scalar('Loss/Train/L1', l1_loss.item(), global_step)
                    writer.add_scalar('Loss/Train/VGG', vgg_loss.item(), global_step)
                    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                    
                    if global_step % 100 == 0:
                        writer.add_images('Images/Input', inputs[0:1, :3], global_step, dataformats='NCHW')
                        writer.add_images('Images/Input_Alpha', inputs[0:1, 3:], global_step, dataformats='NCHW')
                        
                        outputs_vis = outputs[0:1]
                        labels_vis = labels[0:1]
                        
                        writer.add_images('Images/Prediction', outputs_vis, global_step, dataformats='NCHW')
                        writer.add_images('Images/Ground_Truth', labels_vis, global_step, dataformats='NCHW')
                        
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
                avg_val_loss = validate_with_chunks(model, val_loader, criterion, device)
                
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

def validate_with_chunks(model, val_loader, criterion, device, chunk_size=256):
    """使用分块处理的验证函数
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        chunk_size: 分块大小，默认256x256
    """
    model.eval()
    val_loss = 0.0
    val_l1_loss = 0.0
    val_batch_count = 0
    
    with torch.inference_mode():  # 比torch.no_grad()更高效
        for val_inputs, val_labels in val_loader:
            try:
                # 分块处理大分辨率图像
                chunk_loss = 0.0
                chunks = max(1, min(val_inputs.shape[2] // chunk_size, val_inputs.shape[3] // chunk_size))
                
                for i in range(chunks):
                    for j in range(chunks):
                        h_start = i * val_inputs.shape[2] // chunks
                        h_end = (i + 1) * val_inputs.shape[2] // chunks
                        w_start = j * val_inputs.shape[3] // chunks
                        w_end = (j + 1) * val_inputs.shape[3] // chunks
                        
                        chunk_in = val_inputs[:, :, h_start:h_end, w_start:w_end].to(device, non_blocking=True)
                        chunk_label = val_labels[:, :, h_start:h_end, w_start:w_end].to(device, non_blocking=True)
                        
                        with torch.amp.autocast('cuda', 
                                            dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, 
                                            enabled=True): 
                            chunk_out = model(chunk_in)
                            loss = criterion(chunk_out, chunk_label, chunk_in)
                        
                        # 立即转移数据到CPU并释放内存
                        chunk_loss += loss.detach().cpu().item()
                        del chunk_in, chunk_label, chunk_out, loss
                        torch.cuda.empty_cache()
                
                # 计算平均损失
                val_loss += chunk_loss / (chunks * chunks)
                val_batch_count += 1
                
            except Exception as e:
                logging.error(f"验证时出错: {str(e)}")
                continue
    
    return val_loss / val_batch_count if val_batch_count > 0 else float('inf')

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

def load_dataset(processed_dir, batch_size, num_workers=0):
    """
    加载预处理后的数据集
    
    Args:
        processed_dir: 预处理数据目录
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
    """
    # 加载训练集
    train_inputs = np.load(os.path.join(processed_dir, 'train_inputs.npy'), mmap_mode='r')
    train_labels = np.load(os.path.join(processed_dir, 'train_labels.npy'), mmap_mode='r')
    
    # 加载验证集
    val_inputs = np.load(os.path.join(processed_dir, 'val_inputs.npy'), mmap_mode='r')
    val_labels = np.load(os.path.join(processed_dir, 'val_labels.npy'), mmap_mode='r')
    
    # 创建数据集
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_inputs.copy()).float(),  # 转换为float32
        torch.from_numpy(train_labels.copy()).float()   # 转换为float32
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(val_inputs.copy()).float(),    # 转换为float32
        torch.from_numpy(val_labels.copy()).float()     # 转换为float32
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    
    # 设置日志
    log_dir = config.get('base', 'log_dir')
    setup_logger(log_dir)
    
    # 初始化模型
    input_channels = int(config.get('base', 'input_channels'))
    output_channels = int(config.get('base', 'output_channels'))
    model = Unet(input_channels, output_channels).to(device)
    
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
            logging.warning(f"建议将批量大小从{initial_batch_size}调整为{optimal_batch_size}")
            if optimal_batch_size < initial_batch_size:
                logging.warning("当前批量大小可能导致显存溢出")
        
        batch_size = optimal_batch_size
    else:
        batch_size = int(config.get('base', 'batch_size'))
    
    # 完全禁用多进程加载
    num_workers = 0
    
    # 加载数据集
    train_loader, val_loader = load_dataset(
        processed_dir=config['base']['processed_data_dir'],
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # 定义损失函数和优化器
    criterion = CustomLoss(device=device, 
        alpha=float(config.get('base', 'alpha'))
    ).to(device)
    
    # 添加学习率调度器
    initial_learning_rate = float(config.get('base', 'learning_rate'))
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    
    # 添加学习率调度器
    scheduler = None
    if val_loader is not None:
        # 使用ReduceLROnPlateau，在验证损失停止改善时降低学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,        # 每次将学习率降低为原来的一半
            patience=5,        # 5个epoch没有改善就降低学习率
            verbose=True,
            min_lr=1e-6,      # 最小学习率
            eps=1e-8          # 防止混合精度下梯度过小导致的问题
        )
    else:
        # 如果没有验证集，使用余弦退火调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get('base', 'num_epochs')),  # 总epoch数
            eta_min=1e-6,     # 最小学习率
            verbose=True
        )
    
    logging.info(f"初始学习率设置为: {initial_learning_rate}")
    
    # 开始训练
    logging.info(f"Starting training on device: {device}")
    num_epochs = int(config.get('base', 'num_epochs'))
    save_path = os.path.join(config.get('base', 'save_dir'), 'best_model.pth')

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path, scheduler)
    logging.info("Training completed!")

if __name__ == "__main__":
    main()