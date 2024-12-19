import os
import shutil
import random

def organize_dataset(scene_path, output_path, train_ratio=0.8):
    """
    组织数据集，将数据按比例分配到训练集和验证集
    
    Args:
        scene_path: 原始场景数据路径
        output_path: 输出数据路径
        train_ratio: 训练集比例
    """
    # 创建输出目录
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
    
    # 获取所有width文件夹
    input_base = os.path.join(scene_path, 'conditioning_images')
    gt_base = os.path.join(scene_path, 'images')
    width_folders = [f for f in os.listdir(input_base) if f.startswith('width')]
    
    # 收集所有文件对
    all_pairs = []
    for width in width_folders:
        input_dir = os.path.join(input_base, width)
        gt_dir = os.path.join(gt_base, width)
        
        # 获取输入文件列表
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.exr')]
        
        for input_file in input_files:
            base_name = input_file[:-4]  # 移除.exr后缀
            gt_file = base_name + '.png'
            
            if os.path.exists(os.path.join(gt_dir, gt_file)):
                all_pairs.append({
                    'input': os.path.join(input_dir, input_file),
                    'gt': os.path.join(gt_dir, gt_file),
                    'base_name': base_name,
                    'width': width
                })
    
    # 随机打乱并分割数据集
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    # 复制文件到目标目录
    def copy_files(pairs, target_dir):
        for pair in pairs:
            # 构建新的文件名，包含width信息
            new_base_name = f"{pair['width']}_{pair['base_name']}"
            input_target = os.path.join(target_dir, f"{new_base_name}_input.exr")
            gt_target = os.path.join(target_dir, f"{new_base_name}_gt.png")
            
            # 复制文件
            shutil.copy2(pair['input'], input_target)
            shutil.copy2(pair['gt'], gt_target)
            print(f"Copied {new_base_name} to {target_dir}")
    
    # 复制训练集和验证集
    copy_files(train_pairs, os.path.join(output_path, 'train'))
    copy_files(val_pairs, os.path.join(output_path, 'val'))
    
    print(f"\nDataset organization completed!")
    print(f"Total samples: {len(all_pairs)}")
    print(f"Training samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")

def replace_exr_files(data_path, scene_path):
    """
    替换数据集中的exr图像文件
    
    Args:
        data_path: 数据集路径
        scene_path: 新场景数据路径
    """
    # 遍历训练集和验证集
    for subset in ['train', 'val']:
        subset_path = os.path.join(data_path, subset)
        files = [f for f in os.listdir(subset_path) if f.endswith('_input.exr')]
        
        for file in files:
            # 从文件名中提取信息
            base_name = file[:-10]  # 移除'_input.exr'
            width = base_name.split('_')[0]  # 获取width信息
            
            # 构建对应的新exr文件路径
            scene_file = '_'.join(base_name.split('_')[1:]) + '.exr'
            scene_file_path = os.path.join(scene_path, scene_file)
            
            if os.path.exists(scene_file_path):
                # 替换文件
                target_path = os.path.join(subset_path, file)
                shutil.copy2(scene_file_path, target_path)
                print(f"Replaced {file} with new exr file")
            else:
                print(f"Warning: Could not find corresponding file for {file}")
    
    print("\nEXR file replacement completed!")

if __name__ == "__main__":
    scene_path = "/home/kyrie/PCSS_Unet/PCSS-Unet/scene0"
    data_path = "/home/kyrie/PCSS_Unet/PCSS-Unet/data"
    replace_exr_files(data_path, scene_path)
