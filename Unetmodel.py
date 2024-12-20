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
            nn.Conv2d(in_ch, out_ch, 3, padding=1),          #卷积层          
            nn.BatchNorm2d(out_ch),                          #归一化层
            nn.ReLU(inplace=True),                           #激活层
            nn.Conv2d(out_ch, out_ch, 1, padding=0),         #卷积层,使用1x1卷积保持尺寸不变
            nn.BatchNorm2d(out_ch),                          #归一化层
            nn.ReLU(inplace=True)                            #激活层
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        #定义网络模型
        #下采样-》编码
        self.conv1 = DoubleConv(in_ch, 32)  #参数为输入通道数和输出通道数
        self.pool1 = nn.AvgPool2d(2) #根据文章描述，此处2x2的平均池化,而非最大池化
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.AvgPool2d(2)
        #最深层
        self.conv5 = DoubleConv(256, 512)

        # 逆卷积，也可以使用上采样(保证k=stride,stride即上采样倍数)
        self.up6 = nn.Sequential(
            nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=1, padding=0)
        )
        self.conv6 = DoubleConv(256, 256)  # 输入通道从512改为256
        
        self.up7 = nn.Sequential(
            nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=1, padding=0)
        )
        self.conv7 = DoubleConv(128, 128)  # 输入通道从256改为128
        
        self.up8 = nn.Sequential(
            nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=1, padding=0)
        )
        self.conv8 = DoubleConv(64, 64)    # 输入通道从128改为64
        
        self.up9 = nn.Sequential(
            nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0)
        )
        self.conv9 = DoubleConv(32, 32)    # 输入通道从64改为32
        self.conv10 = nn.Conv2d(32, out_ch, 1, padding=0)

    #定义网络前向传播过程
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        #上采样
        up_6 = self.up6(c5)
        # 使用插值来确保尺寸匹配
        if up_6.size() != c4.size():
            up_6 = F.interpolate(up_6, size=c4.size()[2:], mode='bilinear', align_corners=True)
        merge6 = up_6 + c4    # 使用加法替代concatenation
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        if up_7.size() != c3.size():
            up_7 = F.interpolate(up_7, size=c3.size()[2:], mode='bilinear', align_corners=True)
        merge7 = up_7 + c3    # 使用加法替代concatenation
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        if up_8.size() != c2.size():
            up_8 = F.interpolate(up_8, size=c2.size()[2:], mode='bilinear', align_corners=True)
        merge8 = up_8 + c2    # 使用加法替代concatenation
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        if up_9.size() != c1.size():
            up_9 = F.interpolate(up_9, size=c1.size()[2:], mode='bilinear', align_corners=True)
        merge9 = up_9 + c1    # 使用加法替代concatenation
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = torch.sigmoid(c10)
        return out

#创建一个不存在的文件夹
def  makefilepath(folder_path):   
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)
