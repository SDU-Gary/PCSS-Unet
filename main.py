#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
(1)参考文献：UNet网络简单实现
https://blog.csdn.net/jiangpeng59/article/details/80189889

'''
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from  Unetmodel import Unet
from  setdata import LiverDataset
from  setdata import *
from  customLoss import *


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#定义输入数据的预处理模式，因为分为原始图片和研磨图像，所以也分为两种
#image转换为0~1的数据类型
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) #单通道
])

# 此处提供的mask图像是单通道图像，所以mask只需要转换为tensor
y_transforms = transforms.ToTensor()

#model:模型       criterion：损失函数     optimizer：优化器
#dataload：数据   num_epochs：训练轮数
def train_model(model, criterion, optimizer, dataload, num_epochs=5):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1

            #判断是否调用GPU
            inputs = x.to(device)
            labels = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels, inputs) #计算损失值
            loss.backward()
            optimizer.step()#更新所有的参数

            #item（）是得到一个元素张量里面的元素值
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
    #保存模型
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model

#训练模型函数
def train(batch_size,train_path):
    #模型初始化
    model = Unet(4, 1).to(device)
    batch_size = batch_size
    #定义损失函数
    criterion = CustomLoss(alpha=0.9, p=3)
    #定义优化器
    optimizer = optim.Adam(model.parameters())
    #加载训练数据
    liver_dataset = LiverDataset(train_path,transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)

#模型的测试结果
def test(ckptpath,val_path):
    #输入三通道，输出单通道
    model = Unet(4, 1)
    model.load_state_dict(torch.load(ckptpath,map_location='cpu'))
    liver_dataset = LiverDataset(val_path, transform=x_transforms,target_transform=y_transforms)
    #一次加载一张图像
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    #eval函数是将字符串转化为list、dict、tuple，但是字符串里的字符必须是标准的格式，不然会出错
    model.eval()

    import matplotlib.pyplot as plt
    plt.ion()# 打开交互模式
    with torch.no_grad():
        for x, _ in dataloaders:
            y=model(x).sigmoid()
            #a.squeeze(i)   压缩第i维，如果这一维维数是1，则这一维可有可无，便可以压缩
            img_y=torch.squeeze(y).numpy()
            plt.imshow(img_y, cmap='gray')
            plt.pause(0.01)
        plt.show()



if __name__ == '__main__':

    #载入配置模块并调用基础设置节点
    import configparser 
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    config.sections()                               # 获取section节点 
    config.options('base')                          # 获取指定section 的options即该节点的所有键 
    batchsize=config.get("base", "batchsize")       # 获取指定base下的batchsize
    ckptpath=config.get("base", "ckptpath")         # 获取指定base下的ckptpath
    train_path=config.get("base", "train_path")     # 获取指定base下的train_path
    val_path=config.get("base", "val_path")         # 获取指定base下的val_path

    train(batchsize,train_path)                     # 训练
    test(ckptpath,val_path)                         # 测试
