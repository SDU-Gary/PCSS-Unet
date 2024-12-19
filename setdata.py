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

def read_exr(exr_path):
    """读取EXR文件的所有通道"""
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)  # 交换x和y的顺序

    # 读取所有通道
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = file.channels(["R", "G", "B", "A"], FLOAT)
    
    # 转换为numpy数组，并进行归一化
    channels = [np.frombuffer(channel, dtype=np.float32).reshape(size) for channel in channels]
    
    # 对每个通道单独进行归一化
    normalized_channels = []
    for channel in channels:
        # 处理可能的极值和无效值
        channel = np.nan_to_num(channel)  # 将nan替换为0
        if channel.max() - channel.min() > 0:
            channel = (channel - channel.min()) / (channel.max() - channel.min())
        normalized_channels.append(channel)
    
    return normalized_channels

def make_dataset(root):
    imgs = []
    # 获取所有input.exr文件
    input_files = [f for f in os.listdir(root) if f.endswith('_input.exr')]
    
    for input_file in input_files:
        input_path = os.path.join(root, input_file)
        # GT文件是PNG格式
        gt_path = os.path.join(root, input_file.replace('_input.exr', '_gt.png'))
        
        if os.path.exists(gt_path):
            imgs.append((input_path, gt_path))
    return imgs

class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_path, gt_path = self.imgs[index]
        
        # 读取输入EXR文件的四个通道
        input_channels = read_exr(input_path)  # 返回归一化后的通道
        
        # 转换为张量并堆叠
        input_tensors = [torch.from_numpy(channel) for channel in input_channels]
        x = torch.stack(input_tensors, dim=0)  # [4, H, W]
        
        # 读取GT PNG文件
        y = Image.open(gt_path).convert('L')  # 转换为灰度图
        y = transforms.ToTensor()(y)  # 转换为tensor，范围[0,1]
        
        # 应用变换
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        
        # 验证维度
        assert x.size(0) == 4, f"输入应该是4通道,但得到了{x.size(0)}通道"
        assert y.size(0) == 1, f"GT应该是1通道,但得到了{y.size(0)}通道"
        assert x.size(1) == 1080 and x.size(2) == 1920, f"输入图像尺寸应该是1080x1920,但得到了{x.size(1)}x{x.size(2)}"
        assert y.size(1) == 1080 and y.size(2) == 1920, f"GT图像尺寸应该是1080x1920,但得到了{y.size(1)}x{y.size(2)}"
        
        return x, y

    def __len__(self):
        return len(self.imgs)
