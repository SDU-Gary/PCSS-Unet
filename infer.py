import argparse
import torch
import cv2
import numpy as np
import os
import logging

from Unetmodel import Unet
from setdata import read_exr


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for U-net soft shadow generation.")
    parser.add_argument("--input", type=str, required=True, help="Path to input EXR file")
    parser.add_argument("--output", type=str, required=True, help="Path to output grayscale PNG file")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # 配置日志
    logging_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 初始化模型
    model = Unet()
    logging.info(f"正在加载模型权重: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logging.info(f"正在读取输入文件: {args.input}")
    channels = read_exr(args.input)  # 返回4个numpy数组的列表
    
    input_np = np.stack(channels, axis=-1).astype(np.float32)
    input_tensor = torch.from_numpy(input_np).permute(2, 0, 1).unsqueeze(0).to(device)  # 形状: (1, 4, H, W)
    
    # 确保输入尺寸为2的倍数
    H, W = input_tensor.shape[2], input_tensor.shape[3]
    logging.info(f"输入图像尺寸: {H}x{W}")
    
    if H % 2 != 0 or W % 2 != 0:
        H = H - (H % 2)
        W = W - (W % 2)
        logging.info(f"调整输入尺寸为2的倍数: {H}x{W}")
        input_tensor = torch.nn.functional.interpolate(input_tensor, (H, W), mode='bilinear', align_corners=True)
    
    # 执行推理
    logging.info("正在进行模型推理...")
    with torch.inference_mode():  # 比no_grad更优化
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, enabled=True):
            output = model(input_tensor)  # 预期形状: (1, 1, H, W)
    
    # 确保输出在0-1范围内
    output_np = output.squeeze().cpu().numpy()
    
    # 输出统计信息
    logging.info(f"输出图像统计: min={output_np.min():.4f}, max={output_np.max():.4f}, mean={output_np.mean():.4f}")
    
    # 保存为灰度PNG
    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_vis = (output_np * 255).astype(np.uint8)
    cv2.imwrite(output_path, output_vis)
    logging.info(f"推理完成. 输出已保存到 {output_path}")


if __name__ == "__main__":
    main()
