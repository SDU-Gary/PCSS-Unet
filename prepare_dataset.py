#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import os
from setdata import read_exr
from PIL import Image
import logging
import torch
from tqdm import tqdm

def process_files(data_root, output_dir, split='train'):
    """
    处理数据集文件并保存为npy格式
    
    Args:
        data_root: 数据根目录
        output_dir: 输出目录
        split: 数据集划分('train' 或 'val')
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 获取所有input.exr文件
    input_files = [f for f in os.listdir(data_root) if f.endswith('_input.exr')]
    total_files = len(input_files)
    
    inputs_list = []
    labels_list = []
    
    print(f"开始处理{split}数据集，共{total_files}个文件...")
    for input_file in tqdm(input_files, desc=f"处理{split}集"):
        try:
            input_path = os.path.join(data_root, input_file)
            gt_path = os.path.join(data_root, input_file.replace('_input.exr', '_gt.png'))
            
            if not os.path.exists(gt_path):
                logging.warning(f"找不到对应的GT文件: {gt_path}")
                continue
                
            # 读取输入EXR文件
            input_channels = read_exr(input_path)
            input_array = np.stack(input_channels, axis=0)
            
            # 读取标签图像
            label = Image.open(gt_path).convert('L')
            label_array = np.array(label) / 255.0  # 归一化到[0,1]
            
            # 确保输入和标签尺寸匹配
            if input_array.shape[1:] != label_array.shape:
                label = label.resize((input_array.shape[2], input_array.shape[1]), Image.NEAREST)
                label_array = np.array(label) / 255.0
            
            label_array = label_array[np.newaxis, ...]  # 添加通道维度
            
            inputs_list.append(input_array)
            labels_list.append(label_array)
            
        except Exception as e:
            logging.error(f"处理文件 {input_file} 时出错: {str(e)}")
            continue
    
    if not inputs_list:
        raise ValueError(f"没有成功处理任何{split}数据")
    
    # 转换为numpy数组
    inputs = np.stack(inputs_list, axis=0)
    labels = np.stack(labels_list, axis=0)
    
    # 保存为npy文件
    np.save(os.path.join(output_dir, f'{split}_inputs.npy'), inputs)
    np.save(os.path.join(output_dir, f'{split}_labels.npy'), labels)
    
    print(f"{split}数据集处理完成。保存在: {output_dir}")
    print(f"输入形状: {inputs.shape}")
    print(f"标签形状: {labels.shape}")
    
    return inputs.shape[0]  # 返回处理的样本数量

def prepare_dataset(train_root, val_root, output_dir):
    """
    预处理训练集和验证集
    
    Args:
        train_root: 训练集根目录
        val_root: 验证集根目录
        output_dir: 输出目录
    """
    print("开始数据预处理...")
    
    # 处理训练集
    train_samples = process_files(train_root, output_dir, 'train')
    
    # 处理验证集
    val_samples = process_files(val_root, output_dir, 'val')
    
    print("\n数据预处理完成:")
    print(f"训练样本数: {train_samples}")
    print(f"验证样本数: {val_samples}")
    print(f"处理后的数据保存在: {output_dir}")

if __name__ == "__main__":
    train_root = "data/train"  # 训练集目录
    val_root = "data/val"      # 验证集目录
    output_dir = "data/processed"  # 处理后的数据目录
    prepare_dataset(train_root, val_root, output_dir)
