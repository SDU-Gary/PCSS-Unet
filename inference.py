import torch
import numpy as np
import OpenEXR
import Imath
from Unetmodel import Unet
from PIL import Image
import os

def read_exr(exr_path):
    """读取EXR图像文件"""
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
        try:
            file = OpenEXR.InputFile(exr_path)
        except Exception as e:
            raise RuntimeError(f"无法打开EXR文件: {str(e)}")
            
        if not file:
            raise RuntimeError("无法打开EXR文件")
            
        # 获取文件头信息
        try:
            header = file.header()
            dw = header['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            print(f"EXR图像大小: {size}")
            
            # 检查图像尺寸是否合理
            if size[0] <= 0 or size[1] <= 0:
                raise ValueError(f"无效的图像尺寸: {size}")
            if size[0] > 16384 or size[1] > 16384:  # 设置一个合理的最大尺寸
                raise ValueError(f"图像尺寸过大: {size}")
        except Exception as e:
            raise RuntimeError(f"读取EXR文件头信息失败: {str(e)}")

        # 读取RGBA四个通道
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = ['R', 'G', 'B', 'A']
        pixel_data = []
        
        for c in channels:
            try:
                # 检查通道是否存在
                if c not in header['channels']:
                    raise ValueError(f"EXR文件中缺少通道 {c}")
                    
                # 读取通道数据
                data = file.channel(c, FLOAT)
                if not data:
                    raise ValueError(f"无法读取通道 {c} 的数据")
                    
                # 转换为numpy数组
                data = np.frombuffer(data, dtype=np.float32)
                if data is None or len(data) == 0:
                    raise ValueError(f"通道 {c} 数据为空")
                    
                # 检查数据是否包含无效值
                if np.isnan(data).any() or np.isinf(data).any():
                    raise ValueError(f"通道 {c} 包含无效值(NaN或Inf)")
                    
                # 重塑数组
                data = data.reshape(size[1], size[0])
                pixel_data.append(data)
                print(f"通道 {c} 形状: {data.shape}, 范围: [{data.min()}, {data.max()}]")
                
            except Exception as e:
                raise RuntimeError(f"处理通道 {c} 时出错: {str(e)}")

        # 将四个通道堆叠成一个数组
        try:
            exr_img = np.stack(pixel_data, axis=0)
            print(f"最终图像形状: {exr_img.shape}, 范围: [{exr_img.min()}, {exr_img.max()}]")
            
            # 最后的数据验证
            if np.isnan(exr_img).any() or np.isinf(exr_img).any():
                raise ValueError("最终图像数据包含无效值(NaN或Inf)")
                
            return exr_img
            
        except Exception as e:
            raise RuntimeError(f"组合通道数据时出错: {str(e)}")
            
    except Exception as e:
        print(f"读取EXR文件时出错: {str(e)}")
        raise

def save_output(output, save_path):
    """保存输出图像"""
    # 将输出转换为numpy数组并归一化到0-255范围
    output_np = output.squeeze().cpu().numpy()
    output_np = (output_np * 255).astype(np.uint8)
    
    # 创建PIL图像并保存
    output_img = Image.fromarray(output_np)
    output_img.save(save_path)

def process_image(model, input_path, output_path):
    """处理单张图像"""
    try:
        # 读取EXR图像
        print(f"正在读取图像: {input_path}")
        input_img = read_exr(input_path)
        
        # 检查输入图像的尺寸
        if input_img.shape[1] % 16 != 0 or input_img.shape[2] % 16 != 0:
            print(f"警告：输入图像尺寸 ({input_img.shape[1]}, {input_img.shape[2]}) 不是16的倍数")
            print("正在调整图像尺寸...")
            # 计算需要的填充
            pad_h = (16 - input_img.shape[1] % 16) % 16
            pad_w = (16 - input_img.shape[2] % 16) % 16
            # 使用反射填充保持边缘像素
            input_img = np.pad(input_img, ((0,0), (0,pad_h), (0,pad_w)), mode='reflect')
            print(f"调整后的图像尺寸: {input_img.shape}")
        
        # 分别处理每个通道
        print("正在处理各个通道...")
        channels = ['R', 'G', 'B', 'A']
        for i, channel in enumerate(channels):
            channel_data = input_img[i]
            min_val = channel_data.min()
            max_val = channel_data.max()
            print(f"通道 {channel} 原始范围: [{min_val}, {max_val}]")
            
            # 对每个通道单独进行归一化
            if max_val > min_val:
                input_img[i] = (channel_data - min_val) / (max_val - min_val)
            else:
                input_img[i] = np.zeros_like(channel_data)
            
            print(f"通道 {channel} 归一化后范围: [{input_img[i].min()}, {input_img[i].max()}]")
        
        # 转换为torch tensor并添加batch维度
        try:
            input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)
            print(f"输入张量形状: {input_tensor.shape}, 范围: [{input_tensor.min().item()}, {input_tensor.max().item()}]")
            
            # 检查是否有NaN或Inf值
            if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
                raise ValueError("输入数据包含NaN或Inf值")
            
            # 确保数据在正确的设备上
            input_tensor = input_tensor.to(model.device)
            
            # 使用模型进行推理
            print("正在处理图像...")
            with torch.no_grad():
                try:
                    # 清理GPU缓存（如果使用GPU）
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    output = model(input_tensor)
                    print(f"模型输出形状: {output.shape}, 范围: [{output.min().item()}, {output.max().item()}]")
                    
                    # 检查输出是否有效
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        raise ValueError("模型输出包含NaN或Inf值")
                    
                    output = torch.sigmoid(output)
                    print(f"Sigmoid后输出范围: [{output.min().item()}, {output.max().item()}]")
                    
                except RuntimeError as e:
                    print(f"模型推理出错: {str(e)}")
                    print(f"输入张量设备: {input_tensor.device}, 模型设备: {next(model.parameters()).device}")
                    raise
                except Exception as e:
                    print(f"模型推理时发生未知错误: {str(e)}")
                    raise
            
            # 保存输出
            print(f"正在保存输出到: {output_path}")
            save_output(output, output_path)
            print("处理完成!")
            
        except Exception as e:
            print(f"处理张量时出错: {str(e)}")
            raise
            
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def main():
    try:
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # 设置torch为确定性模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 检查模型文件是否存在
        model_path = './checkpoints/best_model.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 创建模型实例
        print("正在加载模型...")
        model = Unet(in_ch=4, out_ch=1)
        
        # 安全加载模型权重
        try:
            # 使用weights_only=True来安全加载模型
            if device.type == 'cuda':
                state_dict = torch.load(model_path, map_location=device)
            else:
                state_dict = torch.load(model_path, map_location='cpu')
                
            model.load_state_dict(state_dict)
            print("模型加载成功")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise

        model.to(device)
        model.eval()
        
        # 保存模型所在设备信息
        model.device = device

        # 创建输出目录
        os.makedirs("outputs", exist_ok=True)

        # 设置输入输出路径
        input_path = "./test.exr"  # 替换为您的输入图像路径
        output_path = "outputs/output.png"
        
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        # 处理图像
        process_image(model, input_path, output_path)

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
