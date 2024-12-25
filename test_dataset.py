#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys
import logging
from setdata import LiverDataset
import torch
from torch.utils.data import DataLoader
import traceback

def test_single_item(dataset, index):
    """测试单个数据项的加载"""
    try:
        logging.info(f"测试加载索引 {index} 的数据")
        x, y = dataset[index]
        logging.info(f"成功加载数据:")
        logging.info(f"输入形状: {x.shape}")
        logging.info(f"标签形状: {y.shape}")
        logging.info(f"输入范围: [{x.min():.4f}, {x.max():.4f}]")
        logging.info(f"标签范围: [{y.min():.4f}, {y.max():.4f}]")
        return True
    except Exception as e:
        logging.error(f"加载数据失败: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def test_dataloader(dataset, batch_size=1):
    """测试数据加载器"""
    try:
        logging.info(f"测试DataLoader，批量大小={batch_size}")
        loader = DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0)
        
        for i, (inputs, labels) in enumerate(loader):
            logging.info(f"批次 {i}:")
            logging.info(f"输入形状: {inputs.shape}")
            logging.info(f"标签形状: {labels.shape}")
            if i >= 2:  # 只测试前几个批次
                break
        return True
    except Exception as e:
        logging.error(f"DataLoader测试失败: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('dataset_test.log')
        ]
    )
    
    try:
        # 从配置文件获取数据集路径
        import configparser
        config = configparser.ConfigParser()
        config.read('config.ini')
        train_dir = config.get('base', 'train_dir')
        
        logging.info(f"测试数据集: {train_dir}")
        
        # 创建数据集
        dataset = LiverDataset(train_dir)
        logging.info(f"数据集大小: {len(dataset)}")
        
        # 测试单个数据项
        success = test_single_item(dataset, 0)
        if not success:
            logging.error("单个数据项测试失败")
            return
        
        # 测试DataLoader
        success = test_dataloader(dataset)
        if not success:
            logging.error("DataLoader测试失败")
            return
        
        logging.info("所有测试通过!")
        
    except Exception as e:
        logging.error(f"测试过程中出错: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
