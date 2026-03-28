# 驾驶员状态感知轨迹预测系统项目记忆

## 项目概述
- 基于EMP模型的驾驶员状态感知轨迹预测系统
- 包含情绪（6分类）和分心行为（10分类）编码器
- 使用Transformer架构进行轨迹编码和解码
- 支持多轨迹预测和评估

## 数据集
### 轨迹数据集（TrajectoryDataset）
- **数据路径**：/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/emp
- **数据集划分**：train, val, test
- **输入维度**：历史轨迹 [seq_len=50, input_dim=4]，包含[x,y,速度差，时间戳]
- **输出维度**：未来轨迹 [future_steps=60, input_dim=4]
- **数据加载**：使用DataLoader，batch_size=256，num_workers=8，pin_memory=True，persistent_workers=True

### 情绪数据集（EmotionDataset）
- **数据路径**：/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/Constructed_Small_sample_0.85
- **数据集划分**：train, val
- **类别**：6个类别（antipathic=0, fear=1, happy=2, neutral=3, sad=4, surprise=5）
- **输入维度**：图像 [3, 224, 224]
- **输出维度**：情绪标签 [batch_size]
- **数据预处理**：Resize(224,224) → RandomHorizontalFlip → ToTensor → Normalize

### 分心数据集（DistractionDataset）
- **数据路径**：/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/SFDDD/images
- **数据集划分**：train, val, test
- **类别**：10个类别（c0-c9，从安全驾驶到和乘客说话）
- **输入维度**：图像 [3, 224, 224]
- **输出维度**：分心行为标签 [batch_size]
- **数据预处理**：Resize(224,224) → RandomHorizontalFlip → RandomRotation → ColorJitter → ToTensor

## 模型结构
### 情绪编码器（EmotionEncoder）
- **输入**：
  - 图像 [batch_size, 3, 224, 224] 或 
  - 轨迹特征 [batch_size, 128]
- **输出**：
  - 均值 mu [batch_size, latent_dim]
  - 对数方差 logvar [batch_size, latent_dim]
  - 采样的潜在向量 z [batch_size, latent_dim]
  - 分类预测 pred [batch_size, num_classes=6]（可选）
- **结构**：
  - CNN编码器（包含增强型块EnhancedBlock和SEBlock）
  - 标签嵌入层
  - 全连接层映射到均值和对数方差
  - 分类头部

### 分心编码器（DistractionEncoder）
- **输入**：
  - 图像 [batch_size, 3, 224, 224] 或 
  - 轨迹特征 [batch_size, 128]
- **输出**：
  - 均值 mu [batch_size, latent_dim]
  - 对数方差 logvar [batch_size, latent_dim]
  - 采样的潜在向量 z [batch_size, latent_dim]
  - 分类预测 pred [batch_size, num_classes=10]（可选）
- **结构**：
  - 卷积特征提取
  - Mamba块（包含通道注意力和空间注意力）
  - 全局平均池化
  - 标签嵌入层
  - 轨迹特征投影层（128→256）
  - 全连接层映射到均值和对数方差
  - 分类头部

### 轨迹编码器/解码器
- **输入**：轨迹数据 [batch_size, num_agents=1, history_steps=50, 4]
- **输出**：预测轨迹 [batch_size, k=6, future_steps=60, 2]
- **结构**：Transformer架构

### 完整模型（FullModel）
- **输入**：
  - 情绪图像 [batch_size, 3, 224, 224]
  - 分心图像 [batch_size, 3, 224, 224]
  - 轨迹数据 [batch_size, num_agents, history_steps, 4]
- **输出**：
  - 预测轨迹 [batch_size, k, future_steps, 2]
  - 情绪潜在表示 [batch_size, latent_dim]
  - 分心潜在表示 [batch_size, latent_dim]
- **结构**：
  - 情绪编码器
  - 分心编码器
  - 轨迹编码器
  - FiLM调制模块
  - 轨迹解码器

## 训练流程
### Stage0：轨迹网络预训练
- **使用数据集**：仅emp数据集
- **加载预训练权重**：
  - 情绪模型：/home/zdx/python_daima/MVim/manuscripts/checkpoints/face/best_model.pth
  - 分心模型：/home/zdx/python_daima/MVim/manuscripts/checkpoints/pose/best_model.pth
- **冻结模块**：情绪编码器、分心编码器、FiLM模块
- **优化目标**：轨迹预测性能
- **模型保存**：只保存最佳和最新模型

### Stage1：情绪和分心编码器训练
- **使用数据集**：仅情绪和分心数据集
- **优化器**：分别使用独立的AdamW优化器
- **学习率调度**：余弦退火学习率调度器
- **损失函数**：交叉熵损失 + KL散度损失
- **模型保存**：保存最佳情绪模型、最佳分心模型和最新模型

### Stage2：FiLM调制训练
- **使用数据集**：仅emp数据集
- **冻结模块**：情绪编码器、分心编码器
- **优化目标**：轨迹预测性能，使用FiLM融合驾驶员状态特征
- **模型保存**：只保存最佳和最新模型



## 损失函数
- **轨迹损失**：smooth L1 loss + 交叉熵损失（用于多轨迹预测）
- **情绪分类损失**：交叉熵损失
- **分心分类损失**：交叉熵损失
- **KL散度损失**：用于VAE正则化
- **正交损失**：用于约束情绪和分心潜在表示的正交性

## 评价指标
- **轨迹预测**：minADE, minFDE, MR
- **情绪分类**：准确率, F1 Score
- **分心分类**：准确率, F1 Score

## 训练配置
- **设备**：单GPU训练（指定GPU 0）
- **批量大小**：256
- **学习率**：2e-4（线性缩放）
- **混合精度训练**：使用torch.cuda.amp.GradScaler
- **数据加载优化**：num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True
- **学习率调度**：余弦退火学习率调度器

## 主要修改和优化
1. **轨迹模型架构**：将轨迹编码器和解码器更改为Transformer架构
2. **训练配置**：从CPU训练优化为单GPU训练
3. **损失计算**：实现minADE和minFDE损失函数用于多轨迹评估
4. **标签处理**：添加标签范围验证和裁剪，解决标签超出范围问题
5. **CUDA优化**：在GPU上禁用确定性算法，避免CuBLAS错误
6. **情绪编码器优化**：使用增强型网络结构（SEBlock和EnhancedBlock）
7. **分心编码器优化**：使用Mamba结构，包含通道注意力和空间注意力模块
8. **批量大小优化**：从32逐步增加到256，提高GPU利用率
9. **学习率优化**：按线性缩放法则调整学习率，从1e-4调整到2e-4
10. **数据加载优化**：添加persistent_workers=True参数，减少每个epoch的启动时间
11. **预训练权重加载**：在Stage0训练前加载情绪和分心模型的预训练权重
12. **模型保存优化**：只保存最佳和最新模型，避免每个epoch都保存
13. **可视化功能**：添加损失曲线和混淆矩阵可视化
14. **日志记录**：添加训练过程和指标的日志记录

## 解决的问题
- 数据集大小不匹配（使用各自数据集的实际大小）
- CUDA设备端断言错误（优化为单GPU训练）
- 标签索引超出范围（添加验证和裁剪）
- 损失计算维度不匹配（更新损失函数）
- 确定性算法导致的CUDA错误（在GPU上禁用）
- 训练启动时间长（添加persistent_workers=True）
- 模型初始化问题（加载预训练权重）
- 损失值过大（优化模型架构和训练策略）
- 模型保存过多（只保存最佳和最新模型）

## 技术创新
- 结合Mamba结构和注意力机制提高特征提取能力
- 多阶段训练策略，逐步优化不同模块
- 独立优化器和学习率调度器，针对不同任务特点调整
- 损失可视化功能，便于分析训练过程
- 预训练权重加载，加速模型收敛
- 轨迹特征作为情绪和分心编码器的输入，实现多模态融合

## 性能指标
- 轨迹预测：minADE和minFDE
- 情绪分类：准确率和F1 Score
- 分心行为分类：准确率和F1 Score
- 模型融合效果：轨迹预测精度提升
- 训练效率：GPU利用率提升到60-80%
- 训练时间：每个epoch启动时间减少

## 预训练权重
- 情绪模型：/home/zdx/python_daima/MVim/manuscripts/checkpoints/face/best_model.pth
- 分心模型：/home/zdx/python_daima/MVim/manuscripts/checkpoints/pose/best_model.pth
