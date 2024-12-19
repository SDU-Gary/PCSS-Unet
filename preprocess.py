import OpenEXR
import Imath
import numpy as np
import os
from tqdm import tqdm
import torch
from pathlib import Path

def read_exr(exr_path):
    """
    读取EXR图片的四个通道
    
    Args:
        exr_path: EXR文件路径
    
    Returns:
        tuple: 包含四个通道的numpy数组
    """
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # 读取所有通道
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = file.channels(["R", "G", "B", "A"], FLOAT)
    
    # 转换为numpy数组
    channels = [np.frombuffer(channel, dtype=np.float32).reshape(size) for channel in channels]
    return tuple(channels)

def process_exr_dataset(input_dir, output_dir, split='train'):
    """
    处理数据集中的所有EXR文件
    
    Args:
        input_dir: 输入EXR文件目录
        output_dir: 输出处理后数据的目录
        split: 数据集划分（train/val/test）
    """
    # 创建输出目录
    output_path = Path(output_dir) / split / 'processed_data'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有EXR文件
    exr_files = list(Path(input_dir).glob('*.exr'))
    
    for exr_file in tqdm(exr_files, desc=f'Processing {split} data'):
        # 读取EXR文件的四个通道
        try:
            r, g, b, a = read_exr(str(exr_file))
            
            # 将四个通道组合成一个数组 (H, W, 4)
            combined = np.stack([r, g, b, a], axis=-1)
            
            # 数据标准化
            combined = (combined - combined.min()) / (combined.max() - combined.min())
            
            # 转换为PyTorch张量并保存
            tensor_data = torch.from_numpy(combined).float()
            output_file = output_path / f"{exr_file.stem}.pt"
            torch.save(tensor_data, output_file)
            
        except Exception as e:
            print(f"Error processing {exr_file}: {str(e)}")

def prepare_dataset(base_dir, output_dir):
    """
    准备完整的数据集
    
    Args:
        base_dir: 原始数据所在的基础目录
        output_dir: 处理后数据的输出目录
    """
    # 处理训练集、验证集和测试集
    for split in ['train', 'val', 'test']:
        input_dir = os.path.join(base_dir, split)
        if os.path.exists(input_dir):
            process_exr_dataset(input_dir, output_dir, split)

if __name__ == '__main__':
    # 示例用法
    base_dir = "dataset"  # 原始EXR文件的基础目录
    output_dir = "dataset"  # 处理后数据的输出目录
    prepare_dataset(base_dir, output_dir)
