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

# 设置设备
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置CUDA内存管理
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清空CUDA缓存
    torch.backends.cudnn.benchmark = True  # 使用cudnn自动寻找最适合当前配置的高效算法
    # 设置较小的网络权重数据类型，以减少内存使用
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

def train_model(model, criterion, optimizer, train_loader, val_loader, config):
    num_epochs = int(config.get('base', 'epochs'))
    save_frequency = int(config.get('base', 'save_frequency'))
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        logging.info('-' * 10)
        
        # 训练阶段
        model.train()
        train_loss = 0
        for step, (inputs, labels) in enumerate(train_loader):
            # 检查输入维度
            if inputs.size(1) != 4:
                logging.error(f"输入通道数错误：期望4通道，实际{inputs.size(1)}通道")
                continue
                
            if labels.size(1) != 1:
                logging.error(f"标签通道数错误：期望1通道，实际{labels.size(1)}通道")
                continue
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            try:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels, inputs)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                if step % 10 == 0:
                    logging.info(f'Step {step} - Train Loss: {loss.item():.4f}')
                    
            except RuntimeError as e:
                logging.error(f"训练步骤出错: {str(e)}")
                continue
                
            # 清理不需要的缓存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels, inputs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f'Epoch {epoch} - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}')
        
        # 保存模型
        if epoch % save_frequency == 0:
            torch.save(model.state_dict(), os.path.join('checkpoints', f'pcss_model_epoch_{epoch}.pth'))
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.get('base', 'best_model_path'))
            logging.info(f'Saved new best model with validation loss: {best_val_loss:.4f}')

def main():
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    
    # 创建必要的目录
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./data/train", exist_ok=True)
    os.makedirs("./data/val", exist_ok=True)
    os.makedirs(config.get('base', 'log_dir'), exist_ok=True)
    
    # 设置日志
    setup_logger(config.get('base', 'log_dir'))
    
    # 设置设备内存限制
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用为80%
    
    # 初始化模型
    input_channels = int(config.get('base', 'input_channels'))
    output_channels = int(config.get('base', 'output_channels'))
    model = Unet(input_channels, output_channels).to(device)
    
    # 定义损失函数和优化器
    criterion = CustomLoss(
        alpha=float(config.get('base', 'alpha')),
        p=int(config.get('base', 'p'))
    )
    learning_rate = float(config.get('base', 'learning_rate'))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 准备数据加载器
    batch_size = int(config.get('base', 'batchsize'))
    train_dataset = LiverDataset(
        config.get('base', 'train_path'),
        transform=x_transforms,
        target_transform=y_transforms
    )
    val_dataset = LiverDataset(
        config.get('base', 'val_path'),
        transform=x_transforms,
        target_transform=y_transforms
    )
    
    # 设置生成器
    g = None
    if torch.cuda.is_available():
        g = torch.Generator(device=device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        generator=g
    )
    
    # 开始训练
    logging.info("Starting training...")
    train_model(model, criterion, optimizer, train_loader, val_loader, config)
    logging.info("Training completed!")

if __name__ == "__main__":
    main()
