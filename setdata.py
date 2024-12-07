#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch.utils.data import Dataset
import PIL.Image as Image
import os

#创建一个列表，存放图像和研磨图像的图像路径
def make_dataset(root):
    imgs=[]
    # 假设每组数据包含5张图像
    n = len(os.listdir(root)) // 5  # 修改为5，因为每组有5张图像
    for i in range(n):
        input_images = [
            os.path.join(root, f"{i:03d}_step1.png"),  # PCSS第一步的中间图
            os.path.join(root, f"{i:03d}_step2.png"),  # PCSS第二步的中间图
            os.path.join(root, f"{i:03d}_step3.png"),  # PCSS第三步的中间图
            os.path.join(root, f"{i:03d}_step4.png")   # PCSS第四步的中间图
        ]
        # GT软阴影图像路径
        gt_image = os.path.join(root, f"{i:03d}_gt.png")
        
        # 确保所有文件都存在
        if all(os.path.exists(img) for img in input_images) and os.path.exists(gt_image):
            imgs.append((input_images, gt_image))
    return imgs


class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform                    #原始图像的预处理
        self.target_transform = target_transform      #研磨图像的预处理

    def __getitem__(self, index):
        input_paths, gt_path = self.imgs[index]
        
        # 读取4张输入图像
        input_images = []
        for path in input_paths:
            img = Image.open(path).convert('L')  # 确保是灰度图
            if self.transform:
                img = self.transform(img)
            input_images.append(img)
        
        # 将4张图像拼接成一个4通道张量
        x = torch.cat(input_images, dim=0)  # 结果维度应该是[4, H, W]
        
        # 读取GT图像
        y = Image.open(gt_path).convert('L')  # 确保是灰度图
        if self.target_transform:
            y = self.target_transform(y)  # 结果维度应该是[1, H, W]
            
        # 验证维度
        assert x.size(0) == 4, f"输入应该是4通道,但得到了{x.size(0)}通道"
        assert y.size(0) == 1, f"GT应该是1通道,但得到了{y.size(0)}通道"
        
        return x, y

    def __len__(self):
        return len(self.imgs)

