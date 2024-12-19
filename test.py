import OpenEXR
import Imath
import numpy as np
import cv2
import os

def check_exr_channels(filename):
    """检查EXR文件的通道数。"""
    try:
        exr_file = OpenEXR.InputFile(filename)
        header = exr_file.header()
        channels = header['channels']
        return len(channels)
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        return 0

def check_dataset_exr_channels(data_path):
    """检查数据集中所有EXR文件的通道数。"""
    print("开始检查数据集中的EXR文件通道数...")
    
    # 遍历训练集和验证集
    for subset in ['train', 'val']:
        subset_path = os.path.join(data_path, subset)
        files = [f for f in os.listdir(subset_path) if f.endswith('_input.exr')]
        
        print(f"\n检查{subset}集中的文件:")
        for file in files:
            file_path = os.path.join(subset_path, file)
            num_channels = check_exr_channels(file_path)
            if num_channels != 4:
                print(f"警告: {file} 包含 {num_channels} 个通道 (应该是4个通道)")
            else:
                print(f"{file}: {num_channels} 通道 √")
    
    print("\nEXR文件检查完成！")

def split_exr_channels(filename):
    """分离 EXR 文件的每个通道并保存为灰度图像。"""
    try:
        exr_file = OpenEXR.InputFile(filename)
        header = exr_file.header()
        channels = header['channels']

        dw = header['dataWindow']
        size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)  # (height, width)

        for channel_name in channels.keys():  # 使用 channels.keys() 而不是 exr_file.channels()
            channel_type = channels[channel_name].type

            if channel_type == Imath.PixelType(Imath.PixelType.FLOAT):
                dtype = np.float32
            elif channel_type == Imath.PixelType(Imath.PixelType.HALF):
                dtype = np.float16
            elif channel_type == Imath.PixelType(Imath.PixelType.UINT):
                dtype = np.uint32
            else:
                print(f"Unsupported channel type for {channel_name}")
                continue

            channel_data = exr_file.channel(channel_name)
            channel = np.frombuffer(channel_data, dtype=dtype)
            channel = channel.reshape(size)

            # 归一化到 0-1 范围
            if channel.dtype != np.uint32:
                channel_min = np.min(channel)
                channel_max = np.max(channel)
                if channel_min != channel_max:
                    channel = (channel - channel_min) / (channel_max - channel_min)
                else:
                    channel = np.zeros_like(channel)

            # 转换为 8 位用于显示和保存
            channel_8bit = (channel * 255).astype(np.uint8)

            # 显示图像
            cv2.imshow(channel_name, channel_8bit)
            
            # 保存图像
            cv2.imwrite(f"{channel_name}.png", channel_8bit)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    data_path = "/home/kyrie/PCSS_Unet/PCSS-Unet/data"
    check_dataset_exr_channels(data_path)