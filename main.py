#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from Unetmodel import Unet
from setdata import LiverDataset
from customLoss import CustomLoss
import configparser
import logging
from datetime import datetime
import torch.cuda.amp as amp
import gc
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn.functional as F

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
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

# 设置随机种子
set_seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置CUDA内存管理
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清空CUDA缓存
    torch.backends.cudnn.benchmark = True  # 使用cudnn自动寻找最适合当前配置的高效算法
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    """训练模型的主循环"""
    import gc
    import torch.cuda.amp as amp
    import logging
    
    # 创建TensorBoard writer
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', current_time)
    writer = SummaryWriter(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

    def log_gpu_memory():
        if device == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB
            logging.info(f'GPU内存使用情况 - 已分配: {allocated:.2f}MB, 已预留: {reserved:.2f}MB')
            # 记录GPU内存使用
            writer.add_scalar('System/GPU_Memory_Allocated_MB', allocated, global_step)
            writer.add_scalar('System/GPU_Memory_Reserved_MB', reserved, global_step)

    scaler = amp.GradScaler()
    best_val_loss = float('inf')
    global_step = 0
    
    # 记录模型结构
    dummy_input = torch.randn(1, 4, 256, 256).to(device)
    writer.add_graph(model, dummy_input)
    
    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_l1_loss = 0.0
            running_vgg_loss = 0.0
            batch_count = 0
            
            # 训练阶段
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                try:
                    # 每5个batch输出一次GPU内存使用情况
                    if batch_idx % 5 == 0:
                        log_gpu_memory()
                    
                    # 清理缓存
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # 数据移动到设备
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    # 检查输入数据
                    if not (torch.isfinite(inputs).all() and torch.isfinite(labels).all()):
                        logging.warning(f"批次 {batch_idx} 包含无效值，跳过")
                        continue
                        
                    # 前向传播
                    optimizer.zero_grad(set_to_none=True)
                    with amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels, inputs)
                        
                        # 获取单独的损失值（假设criterion返回的是总损失）
                        l1_loss = criterion.l1(outputs, labels)
                        vgg_features_out = criterion.get_features(outputs.repeat(1, 3, 1, 1))
                        vgg_features_target = criterion.get_features(labels.repeat(1, 3, 1, 1))
                        vgg_loss = sum(F.mse_loss(out, target) for out, target in zip(vgg_features_out, vgg_features_target))
                    
                    # 反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # 更新统计信息
                    running_loss += loss.item()
                    running_l1_loss += l1_loss.item()
                    running_vgg_loss += vgg_loss.item()
                    batch_count += 1
                    
                    # 记录训练指标
                    writer.add_scalar('Loss/Train/Total', loss.item(), global_step)
                    writer.add_scalar('Loss/Train/L1', l1_loss.item(), global_step)
                    writer.add_scalar('Loss/Train/VGG', vgg_loss.item(), global_step)
                    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                    
                    # 每N个batch记录一次图像
                    if global_step % 100 == 0:
                        # 记录输入图像（只取第一个batch的第一张图片）
                        writer.add_images('Images/Input', inputs[0:1, :3], global_step)  # RGB通道
                        writer.add_images('Images/Input_Alpha', inputs[0:1, 3:], global_step)  # Alpha通道
                        writer.add_images('Images/Prediction', outputs[0:1], global_step)
                        writer.add_images('Images/Ground_Truth', labels[0:1], global_step)
                        # 计算并记录预测与GT的差异
                        diff = torch.abs(outputs - labels)
                        writer.add_images('Images/Prediction_Diff', diff[0:1], global_step)
                    
                    global_step += 1
                    
                    # 定期输出进度
                    if batch_idx % 10 == 0:
                        avg_loss = running_loss / batch_count if batch_count > 0 else 0
                        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {avg_loss:.4f}')
                        
                    # 清理内存
                    del inputs, labels, outputs, loss
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.error(f"GPU内存不足，跳过批次 {batch_idx}")
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    raise e
                except Exception as e:
                    logging.error(f"处理批次 {batch_idx} 时出错: {str(e)}")
                    continue
            
            # 计算并记录每个epoch的平均损失
            epoch_loss = running_loss / batch_count if batch_count > 0 else float('inf')
            epoch_l1_loss = running_l1_loss / batch_count if batch_count > 0 else float('inf')
            epoch_vgg_loss = running_vgg_loss / batch_count if batch_count > 0 else float('inf')
            
            writer.add_scalar('Loss/Train/Epoch_Total', epoch_loss, epoch)
            writer.add_scalar('Loss/Train/Epoch_L1', epoch_l1_loss, epoch)
            writer.add_scalar('Loss/Train/Epoch_VGG', epoch_vgg_loss, epoch)
            
            logging.info(f'Epoch [{epoch+1}/{num_epochs}] 平均训练损失: {epoch_loss:.4f}')
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_l1_loss = 0.0
            val_vgg_loss = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    try:
                        val_inputs = val_inputs.to(device, non_blocking=True)
                        val_labels = val_labels.to(device, non_blocking=True)
                        
                        if not (torch.isfinite(val_inputs).all() and torch.isfinite(val_labels).all()):
                            continue
                            
                        with amp.autocast():
                            val_outputs = model(val_inputs)
                            val_total_loss = criterion(val_outputs, val_labels, val_inputs)
                            
                            # 获取单独的验证损失值
                            val_batch_l1_loss = criterion.l1(val_outputs, val_labels)
                            val_vgg_features_out = criterion.get_features(val_outputs.repeat(1, 3, 1, 1))
                            val_vgg_features_target = criterion.get_features(val_labels.repeat(1, 3, 1, 1))
                            val_batch_vgg_loss = sum(F.mse_loss(out, target) 
                                                   for out, target in zip(val_vgg_features_out, val_vgg_features_target))
                            
                        val_loss += val_total_loss.item()
                        val_l1_loss += val_batch_l1_loss.item()
                        val_vgg_loss += val_batch_vgg_loss.item()
                        val_batch_count += 1
                        
                        del val_inputs, val_labels, val_outputs, val_total_loss
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logging.error(f"验证时出错: {str(e)}")
                        continue
                        
            # 计算并记录验证阶段的平均损失
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
            avg_val_l1_loss = val_l1_loss / val_batch_count if val_batch_count > 0 else float('inf')
            avg_val_vgg_loss = val_vgg_loss / val_batch_count if val_batch_count > 0 else float('inf')
            
            writer.add_scalar('Loss/Val/Total', avg_val_loss, epoch)
            writer.add_scalar('Loss/Val/L1', avg_val_l1_loss, epoch)
            writer.add_scalar('Loss/Val/VGG', avg_val_vgg_loss, epoch)
            
            logging.info(f'Epoch [{epoch+1}/{num_epochs}] 验证损失: {avg_val_loss:.4f}')
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                    }, save_path)
                    logging.info(f'保存最佳模型，验证损失: {best_val_loss:.4f}')
                except Exception as e:
                    logging.error(f"保存模型时出错: {str(e)}")
            
            # 主动进行垃圾回收
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
    except KeyboardInterrupt:
        logging.info("训练被用户中断")
    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}")
        raise
        
    return model

def main():
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # 设置日志
    log_dir = config.get('base', 'log_dir')
    setup_logger(log_dir)
    
    # 设置数据加载器参数
    batch_size = int(config.get('base', 'batch_size'))
    
    # 完全禁用多进程加载
    num_workers = 0
    
    # 创建数据集
    train_dataset = LiverDataset(config.get('base', 'train_dir'), 
                                transform=x_transforms,
                                target_transform=y_transforms)
    
    # 创建顺序索引采样器
    train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            sampler=train_sampler,  # 使用顺序采样器
                            num_workers=num_workers,
                            pin_memory=True)
    
    val_dataset = LiverDataset(config.get('base', 'val_dir'),
                              transform=x_transforms,
                              target_transform=y_transforms)
    
    val_loader = DataLoader(val_dataset,
                          batch_size=batch_size,
                          shuffle=False,  # 验证集不需要打乱
                          num_workers=num_workers,
                          pin_memory=True)
    
    # 初始化模型
    input_channels = int(config.get('base', 'input_channels'))
    output_channels = int(config.get('base', 'output_channels'))
    model = Unet(input_channels, output_channels).to(device)
    
    # 定义损失函数和优化器
    criterion = CustomLoss(
        alpha=float(config.get('base', 'alpha')),
        p=int(config.get('base', 'p'))
    ).to(device)
    
    learning_rate = float(config.get('base', 'learning_rate'))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 开始训练
    logging.info(f"Starting training on device: {device}")
    num_epochs = int(config.get('base', 'num_epochs'))
    save_path = os.path.join(config.get('base', 'save_dir'), 'best_model.pth')
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path)
    logging.info("Training completed!")

if __name__ == "__main__":
    main()