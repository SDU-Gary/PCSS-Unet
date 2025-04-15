import torch
import numpy as np
import OpenEXR
import Imath
from Unetmodel import Unet
from PIL import Image
import os
import argparse
import sys
import torch.nn as nn

def read_exr(exr_path):
    """读取EXR图像文件"""
    file = None
    try:
        # 首先验证文件是否可以打开
        if not os.path.isfile(exr_path):
            raise FileNotFoundError(f"文件不存在: {exr_path}")
            
        # 检查文件大小
        file_size = os.path.getsize(exr_path)
        if file_size == 0:
            raise ValueError(f"文件大小为0: {exr_path}")
        print(f"EXR文件大小: {file_size} bytes")
        
        # 尝试打开文件
        file = OpenEXR.InputFile(exr_path)
        if not file:
            raise RuntimeError("无法打开EXR文件")
            
        # 获取文件头信息
        header = file.header()
        dw = header['dataWindow']
        size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        print(f"EXR图像大小: {size}")
        
        # 检查可用通道
        available_channels = list(header['channels'].keys())
        print(f"可用的通道: {available_channels}")

        # 读取RGBA四个通道
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = ["R", "G", "B", "A"]
        
        # 一次性读取所有通道
        channel_data = file.channels(channels, FLOAT)
        
        normalized_channels = []
        for i, channel_name in enumerate(channels):
            try:
                if channel_data[i] is None:
                    if channel_name == "A":
                        # 如果是Alpha通道不存在，创建全1的通道
                        arr = np.ones(size, dtype=np.float32)
                        print(f"创建默认Alpha通道: shape={size}")
                    else:
                        raise ValueError(f"通道 {channel_name} 数据为空")
                else:
                    # 安全地创建numpy数组
                    arr = np.frombuffer(channel_data[i], dtype=np.float32)
                    if arr is None or arr.size == 0:
                        raise ValueError(f"通道 {channel_name} 数据为空")
                        
                    arr = arr.reshape(size)
                    
                print(f"通道 {channel_name} 形状: {arr.shape}")
                
                # 检查数值范围
                if not np.isfinite(arr).all():
                    print(f"警告：通道 {channel_name} 包含无效值，将替换为有效值")
                    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
                
                print(f"通道 {channel_name} 范围: [{np.min(arr)}, {np.max(arr)}]")
                normalized_channels.append(arr.copy())
                
            except Exception as e:
                print(f"处理通道 {channel_name} 时出错: {str(e)}")
                raise

        # 将四个通道堆叠成一个数组
        exr_img = np.stack(normalized_channels, axis=0)
        print(f"最终图像形状: {exr_img.shape}, 范围: [{exr_img.min()}, {exr_img.max()}]")
        
        return exr_img
            
    except Exception as e:
        print(f"读取EXR文件时出错: {str(e)}")
        raise
        
    finally:
        if file:
            try:
                file.close()
                print("EXR文件已关闭")
            except Exception as e:
                print(f"关闭EXR文件时出错: {str(e)}")

def save_output(output, save_path):
    """保存输出图像，与TensorBoard的可视化格式完全一致"""
    try:
        # 记录输出形状和数据范围
        print(f"保存时的输出形状: {output.shape}")
        print(f"输出范围: [{np.min(output)}, {np.max(output)}]")
        
        # TensorBoard的add_images函数接受[N,C,H,W]格式的数据，值范围在[0,1]之间
        # 我们的输出是[C,H,W]，需要保持相同的处理方式

        # 确保输出在[0,1]范围内，与训练时相同
        output = np.clip(output, 0, 1)
        
        if output.shape[0] == 1:  # 单通道图像
            # 转为[H,W]格式
            output = output.squeeze(0)
            # 转为RGB格式保存
            output_img = Image.fromarray((output * 255).astype(np.uint8), mode='L')
            
        elif output.shape[0] == 3:  # RGB图像
            # 将[3,H,W]转为[H,W,3]
            output = np.transpose(output, (1, 2, 0))
            output_img = Image.fromarray((output * 255).astype(np.uint8), mode='RGB')
            
        elif output.shape[0] == 4:  # RGBA图像
            # 将[4,H,W]转为[H,W,4]
            output = np.transpose(output, (1, 2, 0))
            output_img = Image.fromarray((output * 255).astype(np.uint8), mode='RGBA')
            
        else:
            raise ValueError(f"无法处理的输出通道数: {output.shape[0]}")
        
        # 保存图像
        output_img.save(save_path)
        print(f"图像已保存到: {save_path}")
        
    except Exception as e:
        print(f"保存图像时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_image(model, input_path, output_path):
    """处理单张图像"""
    try:
        # 读取EXR图像
        print(f"正在读取图像: {input_path}")
        input_channels = read_exr(input_path)
        print(f"读取完成，输入数组形状: {len(input_channels)} 通道")
        
        # 将通道转换为tensor，与训练时完全相同的处理方式
        input_tensor = torch.stack([torch.from_numpy(channel) for channel in input_channels], dim=0)
        input_tensor = input_tensor.float()
        
        # 检查输入图像的尺寸
        H, W = input_tensor.shape[1], input_tensor.shape[2]
        if H % 16 != 0 or W % 16 != 0:
            print(f"警告：输入图像尺寸 ({H}, {W}) 不是16的倍数")
            print("正在调整图像尺寸...")
            # 计算需要的填充
            pad_h = (16 - H % 16) % 16
            pad_w = (16 - W % 16) % 16
            # 使用反射填充保持边缘像素
            padder = nn.ReflectionPad2d((0, pad_w, 0, pad_h))
            input_tensor = padder(input_tensor)
            print(f"调整后的图像尺寸: {input_tensor.shape}")
        
        # 添加批次维度 - 与训练时完全相同
        input_tensor = input_tensor.unsqueeze(0)
        print(f"输入张量形状: {input_tensor.shape}")
        print(f"输入范围: [{input_tensor.min().item()}, {input_tensor.max().item()}]")
        
        # 检查是否有NaN或Inf值
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            print("警告：输入数据包含NaN或Inf值，将替换为有效值")
            input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 确保数据在正确的设备上
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        print(f"输入张量已移至设备: {device}")
        
        # 启用与训练时相同的混合精度设置
        with torch.no_grad():
            try:
                # 清理GPU缓存（如果使用GPU）
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 使用与训练时相同的混合精度设置
                with torch.amp.autocast('cuda', 
                                  dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, 
                                  enabled=True):
                    # 进行推理
                    print("正在进行模型推理...")
                    model.eval()  # 确保模型处于评估模式
                    output = model(input_tensor)
                
                print(f"输出形状: {output.shape}")
                print(f"输出范围: [{output.min().item()}, {output.max().item()}]")
                
                # 检查输出是否有效
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print("警告：模型输出包含NaN或Inf值，将替换为有效值")
                    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
                
                # 处理输出 - 与训练时一致
                # CustomLoss 中检查了输出必须经过 Sigmoid，在模型 forward 末尾已有 sigmoid
                
                # 转换为CPU张量并保存
                output_np = output.cpu().numpy()[0]  # 移除批次维度，保持通道维度
                print(f"最终输出形状: {output_np.shape}")
                print(f"最终输出范围: [{output_np.min()}, {output_np.max()}]")
                
            except Exception as e:
                print(f"模型推理时发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
        # 保存输出 - 确保通道顺序正确
        print(f"正在保存输出到: {output_path}")
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        save_output(output_np, output_path)
        print("处理完成!")
        
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def main():
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
        parser.add_argument('--input_path', type=str, required=True, help='Path to input EXR file')
        parser.add_argument('--output_path', type=str, required=True, help='Path to save output image')
        parser.add_argument('--debug', action='store_true', help='开启调试模式，显示更多信息')
        args = parser.parse_args()
        
        # 如果开启调试模式，显示更多信息
        if args.debug:
            print("=== 调试模式已开启 ===")
            print("这将显示训练和推理输出的格式差异")
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # 设置torch为确定性模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 创建并加载模型
        print("正在加载模型...")
        model = Unet()
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 检查checkpoint的格式并相应地加载
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # 如果是完整的checkpoint
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"从epoch {epoch} 加载模型")
                
                # 在调试模式下打印更多信息
                if args.debug and 'optimizer_state_dict' in checkpoint:
                    print("检测到优化器状态")
                    print(f"学习率: {checkpoint['optimizer_state_dict']['param_groups'][0]['lr']}")
                
            else:
                # 如果只是状态字典
                model.load_state_dict(checkpoint)
        else:
            raise ValueError("无法识别的模型文件格式")
            
        model.to(device)
        model.eval()
        
        # 处理图像
        print(f"处理图像: {args.input_path}")
        
        # 如果在调试模式，显示模型的体系结构
        if args.debug:
            print("\n=== 模型结构 ===")
            # 打印模型的主要层和参数数量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"总参数数量: {total_params:,}")
            
            # 提示训练时的处理方式
            print("\n=== 训练时的处理流程 ===")
            print("1. 输入图像通过Unet.rearrange_to_channels进行通道重排")
            print("2. 通过Unet的编码器-解码器结构")
            print("3. 最终输出通过sigmoid激活函数到[0,1]范围")
            print("4. 使用Unet.reconstruct_from_channels重建输出")
            print("5. 输出结果在TensorBoard中直接展示，格式为[N,C,H,W]")
            
        # 执行推理
        process_image(model, args.input_path, args.output_path)
        
        print("处理完成!")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
