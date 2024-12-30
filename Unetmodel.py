#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Unetmodel.py
@Time    :   2021/03/23 20:09:25
@Author  :   Jian Song 
@Contact :   1248975661@qq.com
@Desc    :   None
'''
# here put the import lib

import torch
import torch.nn as nn
from torch import autograd
from visualize import make_dot
import os
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

'''
文件介绍：定义了unet网络模型,
******pytorch定义网络只需要定义模型的具体参数，不需要将数据作为输入定义到网络中。
仅需要在使用时实例化这个网络，然后将数据输入。
******tensorflow定义网络时则需要将输入张量输入到模型中，即用占位符完成输入数据的输入。
'''
#把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),          # 第一个卷积保持通道数不变
            nn.ReLU(inplace=True),                           
            nn.Conv2d(in_ch, out_ch, 1, padding=0),         # 第二个卷积改变通道数
            nn.ReLU(inplace=True)                            
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super(Unet, self).__init__()
        
        # 移除第一层，直接从第二层开始
        # 输入已经从(h × w × 4)重排为(h/2 × w/2 × 16)
        self.conv2 = DoubleConv(16, 32)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = DoubleConv(32, 64)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = DoubleConv(64, 128)
        self.pool4 = nn.AvgPool2d(2)
        
        # 最深层 - 256通道
        self.conv5 = DoubleConv(128, 256)

        # 上采样 - 只使用双线性插值
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = DoubleConv(256, 128)
        
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv7 = DoubleConv(128, 64)
        
        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv8 = DoubleConv(64, 32)
        
        self.up9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv9 = DoubleConv(32, 16)
        
        # 最后一层输出4个通道，用于重构2x2像素块
        self.conv10 = nn.Conv2d(16, 4, 1)

    def rearrange_to_channels(self, x):
        """将输入从(B, 4, H, W)重排为(B, 16, H/2, W/2)"""
        B, C, H, W = x.shape
        # 确保H和W是2的倍数
        assert H % 2 == 0 and W % 2 == 0, "Input dimensions must be even"
        
        # 重排为2x2块
        x = x.view(B, C, H//2, 2, W//2, 2)
        # 将2x2块转换为通道
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        # 合并为16通道
        x = x.view(B, C*4, H//2, W//2)
        return x

    def reconstruct_from_channels(self, x):
        """将输出从4个通道重构为2x2像素块
        输入 x: [B, 4, H, W]
        - 第一个通道通过双线性插值到全分辨率
        - 其余3个通道重排为2x2像素块填充细节
        输出: [B, 1, H*2, W*2]
        """
        B, C, H, W = x.shape
        assert C == 4, "Output must have 4 channels for reconstruction"
        
        # 第一个通道双线性插值到全分辨率
        first_channel = F.interpolate(x[:, 0:1], size=(H*2, W*2), mode='bilinear', align_corners=True)
        
        # 其余三个通道重排为2x2像素块
        other_channels = x[:, 1:]  # [B, 3, H, W]
        
        # 将每个位置的3个值重排为一个2x2块（除了一个位置）
        other_channels = other_channels.permute(0, 2, 3, 1)  # [B, H, W, 3]
        
        # 创建输出tensor
        upsampled = torch.zeros(B, H*2, W*2, device=x.device)
        
        # 填充2x2块
        upsampled[..., ::2, ::2] = other_channels[..., 0]  # 左上
        upsampled[..., ::2, 1::2] = other_channels[..., 1]  # 右上
        upsampled[..., 1::2, ::2] = other_channels[..., 2]  # 左下
        
        # 重塑为所需格式
        upsampled = upsampled.unsqueeze(1)  # [B, 1, H*2, W*2]
        
        # 将细节添加到插值结果中
        return first_channel + upsampled

    def forward(self, x):
        # 使用gradient checkpointing来节省内存
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        # 将输入从(B, 4, H, W)重排为(B, 16, H/2, W/2)
        x = self.rearrange_to_channels(x)
        
        # 编码器部分
        c2 = checkpoint(create_custom_forward(self.conv2), x)
        p2 = checkpoint(create_custom_forward(self.pool2), c2)
        
        c3 = checkpoint(create_custom_forward(self.conv3), p2)
        p3 = checkpoint(create_custom_forward(self.pool3), c3)
        
        c4 = checkpoint(create_custom_forward(self.conv4), p3)
        p4 = checkpoint(create_custom_forward(self.pool4), c4)
        
        # 最深层
        c5 = checkpoint(create_custom_forward(self.conv5), p4)
        
        # 解码器部分
        up_6 = self.up6(c5)
        if up_6.size() != c4.size():
            up_6 = F.interpolate(up_6, size=c4.size()[2:], mode='bilinear', align_corners=True)
        # 先将通道数从256降到128再相加
        up_6 = checkpoint(create_custom_forward(self.conv6), up_6)
        merge6 = up_6 + c4
        # c6 = checkpoint(create_custom_forward(self.conv6), merge6)
        
        up_7 = self.up7(merge6)
        if up_7.size() != c3.size():
            up_7 = F.interpolate(up_7, size=c3.size()[2:], mode='bilinear', align_corners=True)
        # 先将通道数从128降到64再相加
        up_7 = checkpoint(create_custom_forward(self.conv7), up_7)
        merge7 = up_7 + c3
        #c7 = checkpoint(create_custom_forward(self.conv7), merge7)
        
        up_8 = self.up8(merge7)
        if up_8.size() != c2.size():
            up_8 = F.interpolate(up_8, size=c2.size()[2:], mode='bilinear', align_corners=True)
        # 先将通道数从64降到32再相加
        up_8 = checkpoint(create_custom_forward(self.conv8), up_8)
        merge8 = up_8 + c2
        # c8 = checkpoint(create_custom_forward(self.conv8), merge8)
        
        up_9 = self.up9(merge8)
        if up_9.size() != x.size():
            up_9 = F.interpolate(up_9, size=x.size()[2:], mode='bilinear', align_corners=True)
        # 先将通道数从32降到16再相加
        up_9 = checkpoint(create_custom_forward(self.conv9), up_9)
        merge9 = up_9 + x
        # c9 = checkpoint(create_custom_forward(self.conv9), merge9)
        
        c10 = checkpoint(create_custom_forward(self.conv10), merge9)
        
        # 将输出重构为原始分辨率
        out = self.reconstruct_from_channels(c10)
        return out

#创建一个不存在的文件夹
def  makefilepath(folder_path):   
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)
