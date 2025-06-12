# 花卉图像分类项目

这是一个基于 PyTorch 和 ResNet152 的花卉图像分类项目。该项目使用迁移学习技术，通过预训练的 ResNet152 模型对花卉图像进行分类。

## 项目特点

- 使用 ResNet152 作为基础模型
- 采用两阶段训练策略：特征提取和微调
- 支持数据增强和图像预处理
- 使用 TensorBoard 进行训练过程可视化
- 包含完整的训练和预测流程

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- tensorboard

## 安装依赖

```bash
pip install torch torchvision numpy matplotlib tensorboard
```

## 项目结构

```
.
├── train.py          # 训练脚本
├── predict.py        # 预测脚本
├── app.py           # Web应用接口
├── cat_to_name.json # 类别名称映射
└── flower_data/     # 数据集目录
    ├── train/       # 训练集
    └── valid/       # 验证集
```

## 使用方法

### 1. 准备数据

将花卉数据集放置在 `flower_data` 目录下，确保包含 `train` 和 `valid` 两个子目录。

### 2. 训练模型

```bash
python train.py
```

训练过程分为两个阶段：
1. 特征提取阶段：只训练最后一层分类器
2. 微调阶段：训练所有层

### 3. 预测

```bash
python predict.py
```

## 模型架构

- 基础模型：ResNet152
- 分类器：两层全连接层
- 激活函数：ReLU
- 正则化：Dropout ()
- 输出层：LogSoftmax

## 数据增强

训练集使用以下数据增强方法：
- 随机旋转 (±45度)
- 随机水平翻转
- 随机垂直翻转
- 颜色抖动
- 随机灰度转换

## 训练参数

- 批次大小：8
- 特征提取阶段学习率：1e-3
- 微调阶段学习率：1e-4
- 学习率调度：StepLR (step_size=7, gamma=0.1)
- 优化器：Adam

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！ 

## 说明

由于一些权重文件和数据集太大所以没传到github上。权重文件在运行代码时设置下载模式，数据集可以直接在网上找相关花朵数据集即可。