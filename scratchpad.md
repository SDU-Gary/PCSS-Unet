# Lessons

- For website image paths, always use the correct relative path (e.g., 'images/filename.png') and ensure the images directory exists
- For search results, ensure proper handling of different character encodings (UTF-8) for international queries
- Add debug information to stderr while keeping the main output clean in stdout for better pipeline integration
- When using seaborn styles in matplotlib, use 'seaborn-v0_8' instead of 'seaborn' as the style name due to recent seaborn version changes
- When using Jest, a test suite can fail even if all individual tests pass, typically due to issues in suite-level setup code or lifecycle hooks

## Windsurf learned

- For search results, ensure proper handling of different character encodings (UTF-8) for international queries
- Add debug information to stderr while keeping the main output clean in stdout for better pipeline integration
- When using seaborn styles in matplotlib, use 'seaborn-v0_8' instead of 'seaborn' as the style name due to recent seaborn version changes
- Use 'gpt-4o' as the model name for OpenAI's GPT-4 with vision capabilities 

# Scratchpad

## 神经网络实现与论文对比分析

### 网络架构对比

**论文中的特点：**
- 使用5层UNet架构，每层包含1个3x3卷积和1个1x1卷积（不是标准的双3x3卷积）
- 使用双线性插值代替转置卷积进行上采样
- 使用代数和（加法）代替连接层合并跳跃连接
- 使用平均池化代替最大池化
- 移除第一层的跳跃连接以减少噪声

**当前实现：**
- 已实现DoubleConv类使用3x3卷积和1x1卷积
- 使用了双线性插值进行上采样（nn.Upsample）
- 使用算术加法实现跳跃连接（如merge6 = up6 + c4）
- 使用了平均池化（nn.AvgPool2d）
- 直接从第二层开始，第一层通过pixel_unshuffle实现了通道重排

### 损失函数比较

**论文中的特点：**
- 结合像素L1损失和VGG-19感知损失
- 添加特殊的扰动损失以提高时间稳定性
- 扰动损失通过对相机和发射器位置进行微小扰动实现

**当前实现：**
- 已实现CustomLoss结合L1和VGG损失
- VGG损失使用了VGG19预训练模型
- 尚未实现扰动损失用于时间稳定性

### 要点和改进方向

1. **网络架构**：
   - [X] 网络结构基本符合论文中的描述
   - [X] 已使用平均池化和双线性上采样
   - [X] 使用加法而非连接合并跳跃连接

2. **损失函数**：
   - [X] 已实现L1+VGG损失的组合
   - [ ] 考虑添加扰动损失来提高时间稳定性

3. **优化点**：
   - [ ] 对网络的第一层和最后一层特别处理以优化性能
   - [ ] 考虑根据场景特定需求优化网络深度

### 下一步计划

1. 评估当前网络架构性能并与论文中的基准比较
2. 考虑实现扰动损失以提高时间稳定性
3. 检查数据处理流程是否符合论文建议
4. 考虑根据实际阴影大小调整网络深度