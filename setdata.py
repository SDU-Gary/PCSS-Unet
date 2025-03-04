#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import os
import PIL.Image as Image
from torchvision import transforms
import logging
import sys
import traceback
import OpenEXR
import Imath
import gc

# 禁用PIL的调试输出
import PIL
PIL.Image.init()
PIL.Image.PILLOW_VERSION = PIL.__version__  # 兼容性设置
logging.getLogger('PIL').setLevel(logging.WARNING)  # 将PIL的日志级别设置为WARNING

# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dataset_debug.log')
    ]
)

def read_exr(exr_path):
    """读取EXR文件的所有通道，优化内存使用"""
    logging.debug(f"开始读取EXR文件: {exr_path}")
    file = None
    try:
        if not os.path.exists(exr_path):
            raise IOError(f"EXR文件不存在: {exr_path}")
            
        file = OpenEXR.InputFile(exr_path)
        if not file:
            raise IOError(f"无法打开EXR文件: {exr_path}")
            
        dw = file.header()['dataWindow']
        size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        logging.debug(f"EXR文件尺寸: {size}")

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # 预先检查所有通道是否存在
        available_channels = file.header()['channels'].keys()
        logging.debug(f"可用通道: {available_channels}")
        
        # 一次性读取所有通道
        channels = ["R", "G", "B", "A"]
        channel_data = file.channels(channels, FLOAT)
        
        normalized_channels = []
        for i, channel_name in enumerate(channels):
            try:
                if channel_data[i] is None:
                    if channel_name == "A":
                        # 如果是Alpha通道不存在，创建全1的通道
                        arr = np.ones(size, dtype=np.float32)
                        logging.debug(f"创建默认Alpha通道: shape={size}")
                    else:
                        raise ValueError(f"通道 {channel_name} 数据为空")
                else:
                    # 安全地创建numpy数组
                    arr = np.frombuffer(channel_data[i], dtype=np.float32)
                    if arr is None or arr.size == 0:
                        raise ValueError(f"通道 {channel_name} 数据为空")
                        
                    arr = arr.reshape(size)
                    
                logging.debug(f"通道 {channel_name} 形状: {arr.shape}")
                
                # 检查数值范围
                if not np.isfinite(arr).all():
                    logging.warning(f"通道 {channel_name} 包含无效值")
                    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
                
                logging.debug(f"通道 {channel_name} 范围: [{np.min(arr)}, {np.max(arr)}]")
                normalized_channels.append(arr.copy())
                
            except Exception as e:
                logging.error(f"处理通道 {channel_name} 时出错: {str(e)}")
                logging.error(traceback.format_exc())
                raise
            
    except Exception as e:
        logging.error(f"读取EXR文件时出错: {str(e)}")
        logging.error(traceback.format_exc())
        raise
        
    finally:
        if file:
            try:
                file.close()
                logging.debug("EXR文件已关闭")
            except Exception as e:
                logging.error(f"关闭EXR文件时出错: {str(e)}")
    
    if len(normalized_channels) != 4:
        raise ValueError(f"期望4个通道，但得到了 {len(normalized_channels)} 个")
        
    logging.debug("EXR文件读取完成")
    return normalized_channels

def make_dataset(root):
    logging.debug(f"开始扫描数据集目录: {root}")
    imgs = []
    try:
        # 获取所有input.exr文件
        input_files = [f for f in os.listdir(root) if f.endswith('_input.exr')]
        logging.debug(f"找到 {len(input_files)} 个EXR文件")
        
        for input_file in input_files:
            input_path = os.path.join(root, input_file)
            gt_path = os.path.join(root, input_file.replace('_input.exr', '_gt.png'))
            
            if os.path.exists(gt_path):
                imgs.append((input_path, gt_path))
                logging.debug(f"添加数据对: {input_path} - {gt_path}")
            else:
                logging.warning(f"找不到对应的GT文件: {gt_path}")
                
    except Exception as e:
        logging.error(f"扫描数据集时出错: {str(e)}")
        logging.error(traceback.format_exc())
        raise
        
    logging.debug(f"数据集扫描完成，共找到 {len(imgs)} 对有效数据")
    return imgs

class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        if not os.path.exists(root):
            raise ValueError(f"数据集根目录不存在: {root}")
            
        self.imgs = make_dataset(root)
        if len(self.imgs) == 0:
            raise ValueError(f"没有找到有效的数据对")
            
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        
    def _load_data(self, input_path, label_path):
        try:
            # 读取输入EXR文件
            input_channels = read_exr(input_path)
            
            # 将通道数组转换为tensor
            input_tensor = torch.stack([torch.from_numpy(channel) for channel in input_channels], dim=0)
            input_tensor = input_tensor.float()  # 确保是float类型
            input_tensor = input_tensor.requires_grad_(True)  # 确保需要梯度
            input_size = input_tensor.shape[-2:]  # 获取高度和宽度
            logging.debug(f"输入tensor尺寸: {input_tensor.shape}")
            
            # 读取标签图像
            label = Image.open(label_path).convert('L')
            # 调整标签图像大小以匹配输入
            label = label.resize((input_size[1], input_size[0]), Image.NEAREST)
            label_array = np.array(label) / 255.0# 归一化到[0,1]
            label_tensor = torch.from_numpy(label_array).float().unsqueeze(0)  # 添加通道维度
            median_value = torch.median(label_tensor)
            mean_value = torch.mean(label_tensor)
            logging.debug(f"标签图像范围：{label_tensor.min()} - {label_tensor.max()}")
            logging.debug(f"标签图像均值：{mean_value.item()}   标签图像中位数：{median_value.item()}")
            logging.debug(f"标签tensor尺寸: {label_tensor.shape}")
            
            # 验证尺寸匹配
            if input_tensor.shape[-2:] != label_tensor.shape[-2:]:
                raise ValueError(f"输入 ({input_tensor.shape}) 和标签 ({label_tensor.shape}) 尺寸不匹配")
            
            # 应用变换
            if self.transform is not None:
                input_tensor = self.transform(input_tensor)
            if self.target_transform is not None:
                label_tensor = self.target_transform(label_tensor)
            
            # 最终验证
            if input_tensor.shape[-2:] != label_tensor.shape[-2:]:
                raise ValueError(f"变换后输入 ({input_tensor.shape}) 和标签 ({label_tensor.shape}) 尺寸不匹配")
            
            # 确保张量在正确的设备上并且可以计算梯度
            input_tensor = input_tensor.detach().clone()
            input_tensor.requires_grad_(True)  # 明确设置requires_grad
            label_tensor = label_tensor.detach().clone()
            
            return input_tensor, label_tensor
            
        except Exception as e:
            logging.error(f"加载数据时出错: {str(e)}")
            raise IOError(f"读取文件时出错: {str(e)}")
    
    def __getitem__(self, index):
        input_path, label_path = self.imgs[index]
        return self._load_data(input_path, label_path)
        
    def __len__(self):
        return len(self.imgs)

class MmapLiverDataset(Dataset):
    """使用内存映射的数据集实现"""
    def __init__(self, data_dir, transform=None, target_transform=None):
        """
        Args:
            data_dir: 包含inputs.npy和labels.npy的目录路径
            transform: 输入数据的转换
            target_transform: 标签数据的转换
        """
        self.inputs_path = os.path.join(data_dir, 'inputs.npy')
        self.labels_path = os.path.join(data_dir, 'labels.npy')
        
        if not os.path.exists(self.inputs_path) or not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"数据文件不存在。请先运行prepare_dataset.py处理数据。")
            
        # 使用内存映射模式加载数据
        self.inputs = np.load(self.inputs_path, mmap_mode='r')
        self.labels = np.load(self.labels_path, mmap_mode='r')
        
        if self.inputs.shape[0] != self.labels.shape[0]:
            raise ValueError(f"输入数据和标签数量不匹配: {self.inputs.shape[0]} vs {self.labels.shape[0]}")
            
        self.transform = transform
        self.target_transform = target_transform
        
        logging.info(f"使用内存映射加载数据集:")
        logging.info(f"输入形状: {self.inputs.shape}")
        logging.info(f"标签形状: {self.labels.shape}")
        
    def __getitem__(self, index):
        """获取一个数据样本"""
        # 从内存映射数组中读取数据
        input_array = self.inputs[index].astype(np.float32)
        label_array = self.labels[index].astype(np.float32)
        
        # 转换为tensor
        input_tensor = torch.from_numpy(input_array)
        label_tensor = torch.from_numpy(label_array)
        
        # 应用变换
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
        if self.target_transform is not None:
            label_tensor = self.target_transform(label_tensor)
        
        # 确保需要梯度
        input_tensor = input_tensor.detach().clone()
        input_tensor.requires_grad_(True)
        
        return input_tensor, label_tensor
        
    def __len__(self):
        return len(self.inputs)
