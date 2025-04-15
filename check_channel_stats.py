#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import numpy as np
import torch
import argparse
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from setdata import read_exr, MmapLiverDataset

def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('channel_stats_check.log'),
            logging.StreamHandler()
        ]
    )

def analyze_dataset(dataset_path, output_dir, max_samples=None):
    """
    分析数据集中输入图像的通道统计信息
    
    Args:
        dataset_path: 数据集路径（npy文件所在目录）
        output_dir: 输出目录
        max_samples: 最大分析样本数，None表示分析全部
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    try:
        inputs_path = os.path.join(dataset_path, 'train_inputs.npy')
        labels_path = os.path.join(dataset_path, 'train_labels.npy')
        
        if not os.path.exists(inputs_path) or not os.path.exists(labels_path):
            logging.error(f"找不到数据集文件: {inputs_path} 或 {labels_path}")
            return
        
        # 使用内存映射加载数据
        inputs = np.load(inputs_path, mmap_mode='r')
        labels = np.load(labels_path, mmap_mode='r')
        
        logging.info(f"数据集加载成功:")
        logging.info(f"输入形状: {inputs.shape}")
        logging.info(f"标签形状: {labels.shape}")
        
        # 确定分析样本数
        num_samples = len(inputs)
        if max_samples is not None:
            num_samples = min(num_samples, max_samples)
        
        # 创建统计文件
        stats_file = os.path.join(output_dir, 'all_samples_stats.txt')
        
        # 清空之前的统计文件
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"数据集通道统计信息分析\n")
            f.write(f"数据集路径: {dataset_path}\n")
            f.write(f"分析样本数: {num_samples}/{len(inputs)}\n")
            f.write("=" * 80 + "\n\n")
        
        # 存储各通道的汇总统计数据
        channel_stats = {
            'min': [],
            'max': [],
            'mean': [],
            'std': []
        }
        
        # 逐样本分析
        for i in tqdm(range(num_samples), desc="分析样本"):
            input_array = inputs[i].astype(np.float32)
            
            # 记录样本统计信息
            with open(stats_file, 'a', encoding='utf-8') as f:
                f.write(f"\n样本 {i} 的通道统计信息：\n")
                f.write("=" * 50 + "\n")
                
                for c in range(input_array.shape[0]):
                    channel_data = input_array[c]
                    min_val = np.min(channel_data)
                    max_val = np.max(channel_data)
                    mean_val = np.mean(channel_data)
                    std_val = np.std(channel_data)
                    
                    # 存储通道统计信息
                    if len(channel_stats['min']) <= c:
                        channel_stats['min'].append([])
                        channel_stats['max'].append([])
                        channel_stats['mean'].append([])
                        channel_stats['std'].append([])
                    
                    channel_stats['min'][c].append(min_val)
                    channel_stats['max'][c].append(max_val)
                    channel_stats['mean'][c].append(mean_val)
                    channel_stats['std'][c].append(std_val)
                    
                    f.write(f"通道 {c}:\n")
                    f.write(f"  最小值: {min_val:.6f}\n")
                    f.write(f"  最大值: {max_val:.6f}\n")
                    f.write(f"  均值: {mean_val:.6f}\n")
                    f.write(f"  标准差: {std_val:.6f}\n")
                    f.write("-" * 30 + "\n")
        
        # 生成汇总报告
        with open(stats_file, 'a', encoding='utf-8') as f:
            f.write("\n\n" + "=" * 30 + " 汇总统计 " + "=" * 30 + "\n\n")
            for c in range(len(channel_stats['min'])):
                f.write(f"通道 {c} 汇总统计:\n")
                f.write(f"  最小值: 平均={np.mean(channel_stats['min'][c]):.6f}, 范围=[{np.min(channel_stats['min'][c]):.6f}, {np.max(channel_stats['min'][c]):.6f}]\n")
                f.write(f"  最大值: 平均={np.mean(channel_stats['max'][c]):.6f}, 范围=[{np.min(channel_stats['max'][c]):.6f}, {np.max(channel_stats['max'][c]):.6f}]\n")
                f.write(f"  均值: 平均={np.mean(channel_stats['mean'][c]):.6f}, 范围=[{np.min(channel_stats['mean'][c]):.6f}, {np.max(channel_stats['mean'][c]):.6f}]\n")
                f.write(f"  标准差: 平均={np.mean(channel_stats['std'][c]):.6f}, 范围=[{np.min(channel_stats['std'][c]):.6f}, {np.max(channel_stats['std'][c]):.6f}]\n")
                f.write("-" * 60 + "\n")
        
        # 绘制分布图
        plot_channel_distributions(channel_stats, output_dir)
        
        logging.info(f"分析完成，结果保存到: {stats_file}")
        
    except Exception as e:
        logging.error(f"分析数据集时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

def plot_channel_distributions(channel_stats, output_dir):
    """绘制各通道统计量的分布图"""
    stats_types = ['mean', 'std', 'min', 'max']
    
    for stat_type in stats_types:
        plt.figure(figsize=(12, 8))
        
        for c in range(len(channel_stats[stat_type])):
            plt.subplot(2, 2, c+1 if c < 4 else 4)
            plt.hist(channel_stats[stat_type][c], bins=30, alpha=0.7, label=f'Channel {c}')
            plt.title(f'Channel {c} {stat_type} Distribution')
            plt.xlabel(f'{stat_type} Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{stat_type}_分布图.png'), dpi=300)
        plt.close()

def visualize_samples(dataset_path, output_dir, num_samples=5):
    """可视化一些样本图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    inputs_path = os.path.join(dataset_path, 'train_inputs.npy')
    labels_path = os.path.join(dataset_path, 'train_labels.npy')
    
    inputs = np.load(inputs_path, mmap_mode='r')
    labels = np.load(labels_path, mmap_mode='r')
    
    num_samples = min(num_samples, len(inputs))
    
    # 随机选择样本
    indices = np.random.choice(len(inputs), num_samples, replace=False)
    
    for i, index in enumerate(indices):
        input_array = inputs[index].astype(np.float32)
        label_array = labels[index].astype(np.float32)
        
        plt.figure(figsize=(15, 10))
        
        # 绘制输入的各个通道
        for c in range(input_array.shape[0]):
            plt.subplot(2, 3, c+1)
            # 归一化显示，避免极值影响可视化效果
            channel_data = input_array[c]
            vmin, vmax = np.percentile(channel_data, [1, 99])  # 使用1%和99%百分位数避免异常值影响
            plt.imshow(channel_data, cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.title(f'Input Channel {c}')
        
        # 绘制标签
        plt.subplot(2, 3, 5)
        plt.imshow(label_array[0], cmap='gray')
        plt.colorbar()
        plt.title('Label (Shadow Map)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'样本_{index}_可视化.png'), dpi=300)
        plt.close()
    
    logging.info(f"已保存 {num_samples} 个样本的可视化结果")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析数据集通道统计信息')
    parser.add_argument('--dataset_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='channel_stats', help='输出目录')
    parser.add_argument('--max_samples', type=int, default=None, help='最大分析样本数，默认分析全部')
    parser.add_argument('--visualize', action='store_true', help='是否生成样本可视化')
    parser.add_argument('--num_vis_samples', type=int, default=5, help='可视化样本数量')
    
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info(f"开始分析数据集: {args.dataset_path}")
    analyze_dataset(args.dataset_path, args.output_dir, args.max_samples)
    
    if args.visualize:
        logging.info("生成样本可视化...")
        visualize_samples(args.dataset_path, args.output_dir, args.num_vis_samples)
    
    logging.info("分析完成!")
