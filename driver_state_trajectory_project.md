# Driver State-Aware Trajectory Prediction Project

## 1. 项目背景与目标

本项目旨在构建一个**融合驾驶员状态（情绪 + 分心）信息的轨迹预测模型**，用于提升自动驾驶系统对人类驾驶行为的理解与建模能力。

因此，本项目提出：

> **通过学习情绪与分心的潜在表示（latent），调制轨迹预测模型，从而实现更加真实、个性化的驾驶行为建模。**

------------------------------------------------------------------------

## 2. 模型总体结构

模型由三大核心模块组成：

### 2.1 驾驶员状态建模

#### Emotion Encoder

-   输入：驾驶员面部图像
-   输出：
    -   均值：μ_e
    -   方差：σ_e²
    -   latent：z_e
-   建模方式：

z_e \~ q(z_e \| x_e) = N(μ_e, σ_e²)

#### Distraction Encoder

-   输入：驾驶员行为/分心图像
-   输出：
    -   μ_d
    -   σ_d²
    -   z_d
-   建模方式：

z_e \~ q(z_e \| x_e) = N(μ_e, σ_e²)
------------------------------------------------------------------------

### 2.2 轨迹建模

#### Trajectory Encoder

-   输入：历史轨迹 x_t
-   输出：轨迹特征 h_t

#### FiLM 调制模块

融合驾驶员状态：

-   输入：z_e, z_d
-   输出：调制参数 α, β

调制过程：

h' = α ⊙ h_t + β

#### Trajectory Decoder

-   输入：调制后的特征 h'
-   输出：未来轨迹 y_hat

------------------------------------------------------------------------

## 3. 数学建模

### 3.1 Latent建模

情绪：

z_e = μ_e + σ_e \* ε_e\
ε_e \~ N(0, I)

分心：

z_d = μ_d + σ_d \* ε_d\
ε_d \~ N(0, I)

### 3.2 轨迹预测

y_hat = D_t( FiLM( E_t(x_t), z_e, z_d ) )

------------------------------------------------------------------------

## 4. 损失函数设计

L = L_traj + λ1 L_emotion + λ2 L_distraction + λ3 KL_e + λ4 KL_d + λ5 L_orth

### 4.1 轨迹损失

L_traj = MSE(y_hat, y)

### 4.2 分类损失

L_emotion = CrossEntropy\
L_distraction = CrossEntropy

### 4.3 KL散度

KL_e = KL(q(z_e\|x_e) \|\| N(0, I))\
KL_d = KL(q(z_d\|x_d) \|\| N(0, I))

### 4.4 正交约束

L_orth = \|\| z_e\^T z_d \|\|

------------------------------------------------------------------------

## 5. 训练流程

### Stage0：轨迹模型预训练
数据集：emp
权重：预训练的情绪、分心模型权重
训练 Trajectory Encoder + Decoder\
优化 L_traj

### Stage1：状态编码器训练
数据集：情绪、分心数据集
训练 Emotion + Distraction Encoder\
优化 分类 + KL
生成联合权重：stage1_best_combined.pth

### Stage2：FiLM调制训练
数据集：emp
权重：加载stage1_best_combined.pth
冻结情绪和分心编码器，训练 FiLM + Decoder + 轨迹编码器
优化 L_traj

### Stage3：联合微调
数据集：emp
权重：加载stage2_best_combined.pth
训练全模型，优化 L_traj

### Stage4：全流程依次训练

------------------------------------------------------------------------

## 6. 数据集

情绪编码器数据集：[/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/Constructed_Small_sample_0.85](datasets/dataset/Constructed_Small_sample_0.85)
分心编码器数据集：[/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/SFDDD](datasets/dataset/SFDDD)
轨迹数据集：[/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/emp](datasets/dataset/emp)

情绪数据集已经划分为训练集和验证集，/train和 /val，其子目录下的文件夹名称即为类别标签，一共5个类别。
分心数据集已经划分为训练集和测试集，images/train和 images/test，其子目录下的文件夹名称即为类别标签，一共10个类别。
轨迹数据集已经划分为训练集、测试集、验证集，/train、/test、/val。
与情绪、分心数据集有关的配置文件为：[/home/zdx/python_daima/MVim/manuscripts/configs](configs)
与轨迹模型、数据集、训练有关的配置文件为：[/home/zdx/python_daima/MVim/manuscripts/conf](conf)


------------------------------------------------------------------------

## 7. 推理流程

输入：图像 + 历史轨迹\
输出：未来轨迹预测

步骤：

1.  提取 z_e, z_d\
2.  编码轨迹\
3.  FiLM调制\
4.  解码输出

------------------------------------------------------------------------

