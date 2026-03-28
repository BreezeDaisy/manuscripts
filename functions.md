# 损失函数和评价指标文档

## 总损失函数 (TotalLoss)

### 定义
`TotalLoss` 是一个综合损失函数，用于计算模型的总损失，包含轨迹损失、KL散度损失、正交损失和分类损失。

### 计算公式
```
L = λ_traj * L_traj + λ_kl_e * L_kl_e + λ_kl_d * L_kl_d + λ_orth * L_orth + λ_cls_e * L_cls_e + λ_cls_d * L_cls_d
```

### 参数说明
- `λ_traj`: 轨迹损失权重
- `λ_kl_e`: 情绪KL散度损失权重
- `λ_kl_d`: 分心KL散度损失权重
- `λ_orth`: 正交损失权重
- `λ_cls_e`: 情绪分类损失权重
- `λ_cls_d`: 分心分类损失权重

### 输入参数
- `pred`: 预测的轨迹 [batch_size, k, future_steps, 2]
- `target`: 真实的轨迹 [batch_size, future_steps, 5]
- `z_e`: 情绪潜在表示 [batch_size, latent_dim]
- `z_d`: 分心潜在表示 [batch_size, latent_dim]
- `mu_e`: 情绪VAE均值 [batch_size, latent_dim]
- `logvar_e`: 情绪VAE对数方差 [batch_size, latent_dim]
- `mu_d`: 分心VAE均值 [batch_size, latent_dim]
- `logvar_d`: 分心VAE对数方差 [batch_size, latent_dim]
- `pi`: 预测的概率分布 [batch_size, k]
- `pred_e`: 情绪分类预测 [batch_size, num_classes]
- `label_e`: 情绪真实标签 [batch_size]
- `pred_d`: 分心分类预测 [batch_size, num_classes]
- `label_d`: 分心真实标签 [batch_size]

### 输出
- `total_loss`: 总损失
- `traj_loss`: 轨迹损失
- `kl_loss_e`: 情绪KL散度损失
- `kl_loss_d`: 分心KL散度损失
- `orth_loss`: 正交损失
- `cls_loss_e`: 情绪分类损失
- `cls_loss_d`: 分心分类损失
- `min_ade`: 最小平均位移误差
- `min_fde`: 最小最终位移误差
- `mr`: 错过率

## 轨迹损失 (TrajectoryLoss)

### 定义
`TrajectoryLoss` 用于计算轨迹预测的损失，包括回归损失和分类损失，并计算评价指标。

### 计算公式
```
loss = agent_reg_loss + agent_cls_loss
```
- `agent_reg_loss`: 平滑L1损失，计算最佳预测轨迹与真实轨迹的距离
- `agent_cls_loss`: 交叉熵损失，计算预测概率分布与最佳轨迹索引的损失

### 评价指标计算

#### minADE (最小平均位移误差)
```
ade = ||pred - target||_2.mean(-1)
min_ade = ade.min(-1)[0].mean()
```

#### minFDE (最小最终位移误差)
```
fde = ||pred[..., -1, :] - target[..., -1, :]||_2
min_fde = fde.min(-1)[0].mean()
```

#### MR (Miss Rate)
```
missed_pred = fde > miss_threshold
mr = missed_pred.all(-1).float().mean()
```

### 输入参数
- `pred`: 预测的轨迹 [batch_size, k, future_steps, 2]
- `target`: 真实的轨迹 [batch_size, future_steps, 5] 或 None
- `pi`: 预测的概率分布 [batch_size, k]

### 输出
- `loss`: 轨迹损失
- `min_ade`: 最小平均位移误差
- `min_fde`: 最小最终位移误差
- `mr`: 错过率

## 训练阶段损失函数和评价指标

### Stage0: 轨迹网络预训练

#### 数据集
- 只使用emp数据集（轨迹数据）

#### 损失函数
- **主要损失**: 轨迹损失 (`TrajectoryLoss`)
- **计算方式**:
  ```python
  loss, min_ade, min_fde, mr = self.loss_fn.trajectory_loss(pred_dict['y_hat'], y, pred_dict['pi'])
  ```

#### 评价指标
- `minADE`: 最小平均位移误差
- `minFDE`: 最小最终位移误差
- `MR`: 错过率

#### 训练逻辑
1. 加载情绪和分心模型的预训练权重
2. 冻结情绪、分心编码器和FiLM模块
3. 只训练轨迹编码器和解码器
4. 每个epoch计算训练损失和验证损失
5. 保存最佳模型和最新模型

### Stage1: 情绪和分心编码器训练

#### 数据集
- 情绪数据集: Constructed_Small_sample_0.85
- 分心数据集: SFDDD/images

#### 损失函数

##### 情绪编码器
- **损失组成**: 交叉熵损失 + KL散度损失
- **计算方式**:
  ```python
  _, traj_loss, kl_loss_e, kl_loss_d, orth_loss, cls_loss_e, cls_loss_d, _, _, _ = self.loss_fn(
      pred_dict['y_hat'], None, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, 
      pred_dict['pi'], pred_e=pred_e, label_e=label_e
  )
  loss = cls_loss_e + 0.8*kl_loss_e
  ```

##### 分心编码器
- **损失组成**: 交叉熵损失 + KL散度损失
- **计算方式**:
  ```python
  _, traj_loss, kl_loss_e, kl_loss_d, orth_loss, cls_loss_e, cls_loss_d, _, _, _ = self.loss_fn(
      pred_dict['y_hat'], None, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, 
      pred_dict['pi'], pred_d=pred_d, label_d=label_d
  )
  loss = cls_loss_d + kl_loss_d
  ```

#### 评价指标
- **情绪分类**:
  - 准确率 (Accuracy): 使用 `sklearn.metrics.accuracy_score`
  - F1 Score: 使用 `sklearn.metrics.f1_score` (weighted)
- **分心分类**:
  - 准确率 (Accuracy): 使用 `sklearn.metrics.accuracy_score`
  - F1 Score: 使用 `sklearn.metrics.f1_score` (weighted)

#### 训练逻辑
1. 解冻情绪和分心编码器，冻结轨迹编码器和解码器
2. 为情绪和分心模型分别设置优化器和学习率调度器
3. 分别训练情绪和分心编码器
4. 每个epoch计算训练损失和验证损失
5. 保存最佳模型和最新模型
6. 训练完成后，加载最佳模型并生成混淆矩阵

### Stage2: FiLM调制训练

#### 数据集
- 只使用emp数据集（轨迹数据）

#### 损失函数
- **主要损失**: 轨迹损失 (`TrajectoryLoss`)
- **计算方式**:
  ```python
  total_loss_batch, traj_loss, kl_loss_e, kl_loss_d, orth_loss, cls_loss_e, cls_loss_d, min_ade, min_fde, mr = self.loss_fn(pred_dict['y_hat'], y, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_dict['pi'])
  loss = traj_loss
  ```

#### 评价指标
- `minADE`: 最小平均位移误差
- `minFDE`: 最小最终位移误差
- `MR`: 错过率

#### 训练逻辑
1. 解冻FiLM模块和轨迹编码器/解码器，保持情绪和分心编码器冻结
2. 训练FiLM模块和轨迹编码器/解码器
3. 每个epoch计算训练损失和验证损失
4. 保存最佳模型和最新模型

### Stage3: 联合微调训练

#### 数据集
- 只使用emp数据集（轨迹数据）

#### 损失函数
- **总损失**: TotalLoss（包含轨迹损失、KL散度损失和正交损失）
- **计算方式**:
  ```python
  total_loss_batch, traj_loss, kl_loss_e, kl_loss_d, orth_loss, cls_loss_e, cls_loss_d, min_ade, min_fde, mr = self.loss_fn(pred_dict['y_hat'], y, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_dict['pi'])
  loss = total_loss_batch
  ```

#### 评价指标
- `minADE`: 最小平均位移误差
- `minFDE`: 最小最终位移误差
- `MR`: 错过率

#### 训练逻辑
1. 加载Stage2的最佳模型权重
2. 解冻所有模块，进行微调
3. 使用更小的学习率进行训练
4. 每个epoch计算训练损失和验证损失
5. 保存最佳模型和最新模型

## 学习率调度

### Stage0
- 使用余弦退火学习率调度器

### Stage1
- **情绪模型**:
  - 余弦退火调度器，带预热
  - 初始学习率: 1.5 * 基础学习率
  - 最小学习率: 0.01 * 基础学习率
- **分心模型**:
  - 余弦退火调度器，带预热
  - 初始学习率: 1.5 * 基础学习率
  - 最小学习率: 0.01 * 基础学习率

### Stage2
- 余弦退火调度器
- 最小学习率: 0.01 * 基础学习率

### Stage3
- 余弦退火调度器
- 初始学习率: 0.1 * 基础学习率
- 最小学习率: 0.01 * 初始学习率

## 日志记录

每个训练阶段都会将训练过程和重要输出写入日志文件，存储在 `/home/zdx/python_daima/MVim/manuscripts/logs` 目录下：
- Stage0: `stage0_training.log`
- Stage1: `stage1_training.log`
- Stage2: `stage2_training.log`
- Stage3: `stage3_training.log`

日志内容包括：
- 每个epoch的训练损失和验证损失
- 评价指标（如minADE, minFDE, MR, 准确率, F1 Score等）
- 学习率变化
- 模型保存信息

## 模型保存

每个训练阶段都会保存两种模型：
1. **最新模型**: 每个epoch结束后保存
2. **最佳模型**: 基于验证损失最小化保存

保存路径：`/home/zdx/python_daima/MVim/manuscripts/checkpoints`

具体保存的模型文件：
- Stage0: `stage0_latest.pth`, `stage0_best.pth`
- Stage1: `stage1_latest.pth`, `stage1_best_emotion.pth`, `stage1_best_distraction.pth`, `stage1_best_combined.pth`
- Stage2: `stage2_latest.pth`, `stage2_best.pth`
- Stage3: `stage3_latest.pth`, `stage3_best.pth`