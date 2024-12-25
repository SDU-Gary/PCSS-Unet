#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch.utils.data import Dataset
import torch
import OpenEXR
import Imath
import numpy as np
import os
import PIL.Image as Image
from torchvision import transforms
import logging
import sys
import traceback

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
        normalized_channels = []
        
        for channel_name in ["R", "G", "B", "A"]:
            logging.debug(f"正在读取通道: {channel_name}")
            try:
                channel_str = file.channel(channel_name, FLOAT)
                if channel_str is None:
                    raise ValueError(f"通道 {channel_name} 为空")
                    
                # 安全地创建numpy数组
                arr = np.frombuffer(channel_str, dtype=np.float32)
                if arr is None or arr.size == 0:
                    raise ValueError(f"通道 {channel_name} 数据为空")
                    
                arr = arr.reshape(size)
                logging.debug(f"通道 {channel_name} 形状: {arr.shape}")
                
                # 检查数值范围
                if not np.isfinite(arr).all():
                    logging.warning(f"通道 {channel_name} 包含无效值")
                    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
                
                # 移除归一化步骤，直接使用原始值
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
        
        # 禁用缓存以减少内存使用
        self.cache = None

    def _load_data(self, index):
        """实际加载数据的方法"""
        try:
            input_path, gt_path = self.imgs[index]
            
            # 验证文件存在
            if not os.path.exists(input_path) or not os.path.exists(gt_path):
                raise FileNotFoundError(f"输入或GT文件不存在: {input_path}, {gt_path}")
            
            # 读取输入EXR文件
            try:
                input_channels = read_exr(input_path)
                if not input_channels or len(input_channels) != 4:
                    raise ValueError(f"无效的EXR数据")
                    
                # 转换为tensor并立即释放numpy数组
                tensors = []
                for channel in input_channels:
                    tensor = torch.from_numpy(channel.copy())
                    tensors.append(tensor)
                    del channel
                x = torch.stack(tensors, dim=0)
                del tensors
                del input_channels
                
            except Exception as e:
                raise IOError(f"读取EXR文件时出错: {str(e)}")
            
            # 读取GT PNG文件
            try:
                with Image.open(gt_path) as img:
                    # 转换为灰度图并保持原始值
                    img_gray = img.convert('L')
                    y = torch.from_numpy(np.array(img_gray)).unsqueeze(0).float()
                    if not torch.isfinite(y).all():
                        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
            except Exception as e:
                raise IOError(f"读取GT文件时出错: {str(e)}")
            
            # 验证数据维度
            if x.dim() != 3 or x.size(0) != 4:
                raise ValueError(f"无效的输入tensor形状")
            if y.dim() != 3 or y.size(0) != 1:
                raise ValueError(f"无效的GT tensor形状")
            
            # 确保数据类型正确
            x = x.float()
            y = y.float()
            
            # 验证数值范围
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            
            return x, y
            
        except Exception as e:
            logging.error(f"加载数据时出错: {str(e)}")
            logging.error(traceback.format_exc())
            # 返回一个空张量对而不是抛出异常
            return torch.zeros((4, 512, 512), dtype=torch.float32), torch.zeros((1, 512, 512), dtype=torch.float32)

    def __getitem__(self, index):
        try:
            x, y = self._load_data(index)
            
            if self.transform:
                x = self.transform(x)
            if self.target_transform:
                y = self.target_transform(y)
            
            # 最后的安全检查
            if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                logging.warning(f"发现无效值")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
                y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
            
            return x, y
            
        except Exception as e:
            logging.error(f"获取数据时出错: {str(e)}")
            logging.error(traceback.format_exc())
            # 返回一个空张量对而不是抛出异常
            return torch.zeros((4, 512, 512), dtype=torch.float32), torch.zeros((1, 512, 512), dtype=torch.float32)

    def __len__(self):
        return len(self.imgs)
