# PCSS-Unet
 The code reproduction work of Neural Shadow Mapping authored by Christoph Schied, Zhao Dong, et al. and published in the SIGGRAPH'22 conference  

## 训练

### 数据集准备

如果新加入数据集, 需要运行`prepare_dataset.py`生成`.npy`格式的数据集
```bash
python prepare_dataset.py
```

然后运行`calculate_channel_stats.py`进行通道信息统计, 用于后续输入通道标准化
```bash
python calculate_channel_stats.py
```

### 训练

然后运行`main.py`进行训练
```bash
python main.py
```
