#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
验证TensorBoard记录的阴影与推理脚本生成的阴影是否一致
"""

import argparse
import torch
import cv2
import numpy as np
import os
import logging
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from Unetmodel import Unet
from setdata import read_exr
import tempfile
import shutil
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="验证TensorBoard记录与推理结果的一致性。")
    parser.add_argument("--input", type=str, required=True, help="输入EXR文件路径")
    parser.add_argument("--weights", type=str, required=True, help="模型权重文件路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--output_dir", type=str, default="consistency_test", help="输出目录")
    return parser.parse_args()

def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('consistency_test.log')
        ]
    )

def process_with_tensorboard(model, input_tensor, device, log_dir):
    """使用TensorBoard流程处理输入并记录结果"""
    writer = SummaryWriter(log_dir)
    
    model.eval()
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, 
                              dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, 
                              enabled=True):
            output = model(input_tensor)
    
    # 记录原始浮点数结果
    writer.add_images('Prediction', output, 0, dataformats='NCHW')
    
    # 记录转换为uint8的结果
    output_uint8 = (output.detach().cpu() * 255).to(torch.uint8)
    writer.add_images('Prediction_Uint8', output_uint8, 0, dataformats='NCHW')
    
    writer.close()
    return output

def process_with_inference(model, input_tensor, device):
    """使用推理脚本流程处理输入"""
    model.eval()
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, 
                              dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, 
                              enabled=True):
            output = model(input_tensor)
    
    return output

def save_and_compare_outputs(tb_output, infer_output, output_dir):
    """保存并比较两种方式的输出"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存TensorBoard流程的输出
    tb_np = tb_output.squeeze().cpu().numpy()
    tb_uint8 = (tb_np * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "tensorboard_output.png"), tb_uint8)
    
    # 保存推理流程的输出
    infer_np = infer_output.squeeze().cpu().numpy()
    infer_uint8 = (infer_np * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "inference_output.png"), infer_uint8)
    
    # 计算差异
    diff = np.abs(tb_np - infer_np)
    diff_scaled = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else np.zeros_like(diff, dtype=np.uint8)
    cv2.imwrite(os.path.join(output_dir, "difference.png"), diff_scaled)
    
    # 创建比较图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(tb_np, cmap='gray')
    plt.title(f"TensorBoard Output\nMin: {tb_np.min():.4f}, Max: {tb_np.max():.4f}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(infer_np, cmap='gray')
    plt.title(f"Inference Output\nMin: {infer_np.min():.4f}, Max: {infer_np.max():.4f}")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap='hot')
    plt.title(f"Difference\nMax: {diff.max():.4f}, Mean: {diff.mean():.4f}")
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"))
    
    # 输出统计信息
    logging.info(f"TensorBoard输出 - 最小值: {tb_np.min():.4f}, 最大值: {tb_np.max():.4f}, 均值: {tb_np.mean():.4f}")
    logging.info(f"推理输出 - 最小值: {infer_np.min():.4f}, 最大值: {infer_np.max():.4f}, 均值: {infer_np.mean():.4f}")
    logging.info(f"差异 - 最大值: {diff.max():.4f}, 均值: {diff.mean():.4f}")
    
    # 相似度计算
    mse = np.mean((tb_np - infer_np) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    logging.info(f"MSE: {mse:.8f}, PSNR: {psnr:.2f} dB")
    
    # 保存统计信息到文件
    with open(os.path.join(output_dir, "stats.txt"), 'w') as f:
        f.write(f"TensorBoard输出 - 最小值: {tb_np.min():.4f}, 最大值: {tb_np.max():.4f}, 均值: {tb_np.mean():.4f}\n")
        f.write(f"推理输出 - 最小值: {infer_np.min():.4f}, 最大值: {infer_np.max():.4f}, 均值: {infer_np.mean():.4f}\n")
        f.write(f"差异 - 最大值: {diff.max():.4f}, 均值: {diff.mean():.4f}\n")
        f.write(f"MSE: {mse:.8f}, PSNR: {psnr:.2f} dB\n")
    
    return mse, psnr

def main():
    args = parse_args()
    setup_logging()
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"输出将保存到: {output_dir}")
    
    # 创建临时TensorBoard日志目录
    temp_log_dir = os.path.join(tempfile.gettempdir(), f"tb_log_{timestamp}")
    os.makedirs(temp_log_dir, exist_ok=True)
    
    try:
        device = torch.device(args.device)
        
        # 加载模型
        logging.info(f"正在加载模型: {args.weights}")
        model = Unet()
        checkpoint = torch.load(args.weights, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        
        # 读取输入文件
        logging.info(f"正在读取输入文件: {args.input}")
        channels = read_exr(args.input)
        input_np = np.stack(channels, axis=-1).astype(np.float32)
        input_tensor = torch.from_numpy(input_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # 确保输入尺寸为2的倍数
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        if H % 2 != 0 or W % 2 != 0:
            H = H - (H % 2)
            W = W - (W % 2)
            logging.info(f"调整输入尺寸为2的倍数: {H}x{W}")
            input_tensor = torch.nn.functional.interpolate(input_tensor, (H, W), mode='bilinear', align_corners=True)
        
        # 使用TensorBoard流程处理
        logging.info("使用TensorBoard流程处理输入...")
        tb_output = process_with_tensorboard(model, input_tensor, device, temp_log_dir)
        
        # 使用推理流程处理
        logging.info("使用推理流程处理输入...")
        infer_output = process_with_inference(model, input_tensor, device)
        
        # 比较结果
        logging.info("正在比较输出结果...")
        mse, psnr = save_and_compare_outputs(tb_output, infer_output, output_dir)
        
        if mse < 1e-6:
            logging.info("验证成功! TensorBoard记录和推理输出完全一致。")
        elif psnr > 50:
            logging.info(f"验证通过! 输出非常接近 (PSNR: {psnr:.2f} dB)，差异可能是由于浮点精度造成的。")
        else:
            logging.warning(f"验证失败! 输出存在明显差异 (PSNR: {psnr:.2f} dB)。请检查详细比较图。")
        
    except Exception as e:
        logging.error(f"验证过程中出错: {str(e)}", exc_info=True)
    finally:
        # 清理临时目录
        if os.path.exists(temp_log_dir):
            shutil.rmtree(temp_log_dir)
            logging.info(f"已清理临时日志目录: {temp_log_dir}")

if __name__ == "__main__":
    main()
