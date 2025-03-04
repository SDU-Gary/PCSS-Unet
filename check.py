import os
import numpy as np
import OpenEXR
import Imath
import cv2
from pathlib import Path

def read_exr(exr_path):
    """读取EXR文件的所有通道"""
    print(f"开始读取EXR文件: {exr_path}")
    
    try:
        print("打开EXR文件...")
        file = OpenEXR.InputFile(exr_path)
        if not file:
            raise IOError(f"无法打开EXR文件: {exr_path}")
        
        print("获取文件头信息...")
        dw = file.header()['dataWindow']
        size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        print(f"图像尺寸: {size}")

        # 一次性读取所有通道
        print("一次性读取所有通道...")
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels_data = file.channels(["R", "G", "B", "A"], FLOAT)
        
        channels = []
        for i, name in enumerate(["R", "G", "B", "A"]):
            print(f"处理{name}通道...")
            if channels_data[i] is None and name == "A":
                # 如果是Alpha通道且不存在，创建全1的通道
                channel = np.ones(size, dtype=np.float32)
                print(f"创建默认Alpha通道")
            else:
                channel = np.frombuffer(channels_data[i], dtype=np.float32).reshape(size)
            channels.append(channel)
            print(f"{name}通道处理完成")
        
        file.close()
        print("EXR文件读取完成")
        return channels
        
    except Exception as e:
        print(f"读取EXR文件时出错: {str(e)}")
        raise

def check_data_ranges(exr_path, label_path):
    """检查EXR和标签图像的数值范围"""
    # 读取和分析EXR文件
    print(f"\nEXR文件: {os.path.basename(exr_path)}")
    print("-" * 50)
    
    try:
        print("开始读取EXR文件...")  # 新增日志
        channels = read_exr(exr_path)
        print("EXR文件读取完成")  # 新增日志
        
        for i, name in enumerate(["R", "G", "B", "A"]):
            print(f"\n处理{name}通道...")  # 新增日志
            channel = channels[i]
            print(f"\n{name}通道:")
            print(f"最小值: {np.min(channel):.6f}")
            print(f"最大值: {np.max(channel):.6f}")
            print(f"均值: {np.mean(channel):.6f}")
            print(f"标准差: {np.std(channel):.6f}")

        print("\n开始读取标签图像...")  # 新增日志
        # 使用OpenCV读取标签图像
        label_array = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label_array is None:
            raise ValueError(f"无法读取标签图像: {label_path}")
        
        print(f"\n标签图像: {os.path.basename(label_path)}")
        print("-" * 50)
        
        print("\n原始标签:")
        print(f"最小值: {np.min(label_array)}")
        print(f"最大值: {np.max(label_array)}")
        print(f"均值: {np.mean(label_array):.2f}")
        
        # 归一化后的标签
        label_norm = label_array.astype(np.float32) / 255.0
        print("\n归一化后标签:")
        print(f"最小值: {np.min(label_norm):.6f}")
        print(f"最大值: {np.max(label_norm):.6f}")
        print(f"均值: {np.mean(label_norm):.6f}")
        
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        raise

def visualize_channels(exr_path, label_path):
    """可视化EXR的四个通道和标签图像"""
    # 读取EXR文件的通道
    channels = read_exr(exr_path)
    
    # 创建窗口
    cv2.namedWindow('Channels', cv2.WINDOW_NORMAL)
    
    # 准备显示图像
    display_images = []
    
    # 处理每个通道
    for i, (name, channel) in enumerate(zip(["R", "G", "B", "A"], channels)):
        # 归一化到0-255范围用于显示
        normalized = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        # 转换为uint8类型
        gray_img = normalized.astype(np.uint8)
        # 转换为三通道图像以便添加文字
        display_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        
        # 添加通道名称
        text_color = (255, 255, 255)  # 白色文字
        cv2.putText(display_img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, text_color, 2, cv2.LINE_AA)
        
        display_images.append(display_img)
    
    # 读取标签图像 - 保持原始黑白格式
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise ValueError(f"无法读取标签图像: {label_path}")
    
    # 创建三通道的标签图显示，但保持黑白
    label_display = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
    cv2.putText(label_display, 'Label', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    display_images.append(label_display)
    
    # 将所有图像拼接在一起
    # 创建两行，每行3张图
    row1 = np.hstack(display_images[:3])  # RGB通道
    row2 = np.hstack([display_images[3], display_images[4], 
                      np.zeros_like(display_images[0])])  # Alpha通道和标签
    
    # 垂直拼接两行
    final_display = np.vstack([row1, row2])
    
    # 调整窗口大小
    cv2.resizeWindow('Channels', final_display.shape[1] // 2, final_display.shape[0] // 2)
    
    # 显示图像
    cv2.imshow('Channels', final_display)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compare_conversion_methods(label_path):
    """比较不同的图像转换方法"""
    from PIL import Image
    import torch
    import matplotlib.pyplot as plt

    print(f"\n比较图像转换方法: {os.path.basename(label_path)}")
    print("-" * 50)

    # 方法1：使用convert('L')
    img_l = Image.open(label_path).convert('L')
    array_l = np.array(img_l, dtype=np.float32) / 255.0
    tensor_l = torch.from_numpy(array_l).unsqueeze(0)

    # 方法2：直接读取
    img_rgb = Image.open(label_path)
    array_rgb = np.array(img_rgb, dtype=np.float32)
    if len(array_rgb.shape) == 3:
        array_rgb = array_rgb.mean(axis=2)
    array_rgb = array_rgb / 255.0
    tensor_rgb = torch.from_numpy(array_rgb).unsqueeze(0)

    # 打印统计信息
    print("\n方法1 (convert('L')):")
    print(f"形状: {tensor_l.shape}")
    print(f"最小值: {tensor_l.min().item():.6f}")
    print(f"最大值: {tensor_l.max().item():.6f}")
    print(f"均值: {tensor_l.mean().item():.6f}")

    print("\n方法2 (RGB平均):")
    print(f"形状: {tensor_rgb.shape}")
    print(f"最小值: {tensor_rgb.min().item():.6f}")
    print(f"最大值: {tensor_rgb.max().item():.6f}")
    print(f"均值: {tensor_rgb.mean().item():.6f}")

    # 可视化比较
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(array_l, cmap='gray')
    plt.title("Method 1: convert('L')")
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(array_rgb, cmap='gray')
    plt.title("Method 2: RGB Mean")
    plt.colorbar()
    
    plt.subplot(133)
    diff = np.abs(array_l - array_rgb)
    plt.imshow(diff, cmap='hot')
    plt.title("Absolute Difference")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def main():
    # 获取当前目录
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data" / "train"
    
    # 获取第一对图像
    exr_files = list(data_dir.glob("*_input.exr"))
    if not exr_files:
        print("未找到EXR文件")
        return
        
    exr_path = str(exr_files[0])
    label_path = exr_path.replace("_input.exr", "_gt.png")
    
    if not os.path.exists(label_path):
        print(f"找不到对应的标签文件: {label_path}")
        return
    
    # 运行原有的检查
    print("\n=== 检查数据范围 ===")
    check_data_ranges(exr_path, label_path)
    
    print("\n=== 比较转换方法 ===")
    compare_conversion_methods('./Train/Bistro/images/Bistro_Width4Camera3Light0.png')
    
    print("\n=== 可视化通道 ===")
    visualize_channels(exr_path, label_path)

if __name__ == "__main__":
    main()
