import torch
import torch.nn as nn
from torch import autograd
from visualize import make_dot
import os
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging

'''
文件介绍：定义了unet网络模型,
******pytorch定义网络只需要定义模型的具体参数，不需要将数据作为输入定义到网络中。
仅需要在使用时实例化这个网络，然后将数据输入。
******tensorflow定义网络时则需要将输入张量输入到模型中，即用占位符完成输入数据的输入。
'''
#把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.InstanceNorm2d(in_ch),  # 使用实例归一化
            nn.LeakyReLU(0.2, inplace=False),
            
            nn.Conv2d(in_ch, out_ch, 1),  # 1x1卷积改变通道数
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=False),
            
            # nn.Dropout2d(0.1)  # 添加Dropout防止梯度消失
        )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super().__init__()
        # 删除原conv1，直接从第二层开始（输入已是16通道）
        self.conv2 = DoubleConv(16, 32)  # 第二层
        self.pool2 = nn.AvgPool2d(2)
        
        self.conv3 = DoubleConv(32, 64)  # 第三层
        self.pool3 = nn.AvgPool2d(2)
        
        self.conv4 = DoubleConv(64, 128)  # 第四层
        self.pool4 = nn.AvgPool2d(2)
        
        self.conv5 = DoubleConv(128, 256)  # 最深层
        
        # 解码器部分
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = DoubleConv(256, 128)
        
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv7 = DoubleConv(128, 64)
        
        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv8 = DoubleConv(64, 32)
        
        self.up9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv9 = DoubleConv(32, 16)
        
        self.conv10 = nn.Conv2d(16, 4, 1)  # 最终输出4通道

    def rearrange_to_channels(self, x):
        """使用pixel_unshuffle进行输入重排"""
        return F.pixel_unshuffle(x, downscale_factor=2)  # [B,4,H,W] → [B,16,H/2,W/2]

    def reconstruct_from_channels(self, x):
        """使用pixel_shuffle实现可导的2*2像素块重构
        输入x的4个通道分别对应:
        - channel 0: 左上角像素
        - channel 1: 右上角像素
        - channel 2: 左下角像素
        - channel 3: 右下角像素
        """
        B, C, H, W = x.shape
        
        # 重新排列通道顺序以匹配pixel_shuffle的期望输出位置
        # pixel_shuffle会将通道按照如下顺序重排：
        # [左上, 右上, 左下, 右下] -> 2x2网格
        x_shuffled = x  # 通道顺序已经正确，无需调整
        
        # 使用pixel_shuffle进行上采样重构
        # 这是一个完全可导的操作
        output = F.pixel_shuffle(x_shuffled, upscale_factor=2)  # [B,1,H*2,W*2]
        
        return output

    def forward(self, x):
        # 将输入从(B, 4, H, W)重排为(B, 16, H/2, W/2)

        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            H = H - (H % 2)
            W = W - (W % 2)
            x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)

        x = self.rearrange_to_channels(x)
        
        # 编码器部分（从第二层开始）
        c2 = self.conv2(x)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        
        # 最深层使用梯度检查点
        def conv5_block(x):
            return self.conv5(x)
        c5 = checkpoint(conv5_block, p4, use_reentrant=False)
        
        def _upsample_and_match(src, target):
            return F.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=True)
        
        # Layer6
        up6 = self.up6(c5)
        up6 = _upsample_and_match(up6, c4)
        up6 = self.conv6(up6)
        merge6 = up6 + c4
        
        # Layer7
        up7 = self.up7(merge6)
        up7 = _upsample_and_match(up7, c3)
        up7 = self.conv7(up7)
        merge7 = up7 + c3
        
        # Layer8
        up8 = self.up8(merge7)
        up8 = _upsample_and_match(up8, c2)
        up8 = self.conv8(up8)
        merge8 = up8 + c2
        
        # 最终输出
        up9 = self.up9(merge8)
        up9 = _upsample_and_match(up9, x)  # 匹配输入重排后的尺寸
        up9 = self.conv9(up9)
        c10 = self.conv10(up9)
        
        c10 = torch.sigmoid(c10)
        # 重构输出
        out = self.reconstruct_from_channels(c10)
        return out

#创建一个不存在的文件夹
def  makefilepath(folder_path):   
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)
