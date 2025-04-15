#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import numpy as np
import torch
import argparse
import logging
import json
from tqdm import tqdm

def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dataset_stats.log'),
            logging.StreamHandler()
        ]
    )

def calculate_dataset_stats(dataset_path, save_path=None):
    """
    计算数据集的通道均值和标准差
    
    Args:
        dataset_path: 包含train_inputs.npy的目录路径
        save_path: 保存统计数据的路径，默认与dataset_path相同
    
    Returns:
        dict: 包含均值和标准差的字典
    """
    if save_path is None:
        save_path = dataset_path
    
    inputs_path = os.path.join(dataset_path, 'train_inputs.npy')
    
    if not os.path.exists(inputs_path):
        logging.error(f"找不到训练数据文件: {inputs_path}")
        return None
    
    try:
        logging.info(f"正在加载数据集: {inputs_path}")
        # 使用内存映射加载数据，以处理大型数据集
        inputs = np.load(inputs_path, mmap_mode='r')
        
        logging.info(f"数据集形状: {inputs.shape}")
        num_samples = inputs.shape[0]
        num_channels = inputs.shape[1]
        
        # 初始化存储均值和标准差的数组
        means = np.zeros(num_channels, dtype=np.float64)
        squared_sums = np.zeros(num_channels, dtype=np.float64)
        
        # 计算每个通道的均值
        logging.info("计算通道均值...")
        # 首先计算所有样本每个通道的总和
        for i in tqdm(range(num_samples)):
            sample = inputs[i].astype(np.float32)
            for c in range(num_channels):
                # 累加每个通道的像素值
                means[c] += np.sum(sample[c])
        
        # 计算均值：总和除以像素总数
        pixel_count = inputs.shape[2] * inputs.shape[3]
        means = means / (num_samples * pixel_count)
        
        # 计算每个通道的标准差
        logging.info("计算通道标准差...")
        for i in tqdm(range(num_samples)):
            sample = inputs[i].astype(np.float32)
            for c in range(num_channels):
                # 累加每个像素与均值的差的平方
                squared_diff = (sample[c] - means[c]) ** 2
                squared_sums[c] += np.sum(squared_diff)
        
        # 计算标准差：平方和的平均值的平方根
        stds = np.sqrt(squared_sums / (num_samples * pixel_count))
        
        # 创建结果字典
        stats = {
            'means': means.tolist(),
            'stds': stds.tolist()
        }
        
        # 保存结果为NumPy文件
        stats_npy_path = os.path.join(save_path, 'train_stats.npy')
        np.save(stats_npy_path, stats)
        logging.info(f"统计数据已保存为NumPy文件: {stats_npy_path}")
        
        # 同时保存为JSON格式，方便查看
        stats_json_path = os.path.join(save_path, 'train_stats.json')
        with open(stats_json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        logging.info(f"统计数据已保存为JSON文件: {stats_json_path}")
        
        # 打印结果
        logging.info(f"数据集通道均值: {means}")
        logging.info(f"数据集通道标准差: {stds}")
        
        return stats
    
    except Exception as e:
        logging.error(f"计算统计数据时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算数据集的通道均值和标准差')
    parser.add_argument('--dataset_path', type=str, required=True, help='包含train_inputs.npy的目录路径')
    parser.add_argument('--save_path', type=str, default=None, help='保存统计数据的路径，默认与dataset_path相同')
    
    args = parser.parse_args()
    
    setup_logging()
    calculate_dataset_stats(args.dataset_path, args.save_path)
