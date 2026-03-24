# 导入必要的库
import torch

# 模型配置
model_config = {
    # 情绪编码器配置
    'latent_dim': 32,  # VAE潜在空间维度
    'emotion_num_classes': 6,  # 情绪类别数量
    'distraction_num_classes': 10,  # 分心类别数量
    
    # 轨迹编码器配置
    'trajectory_input_dim': 5,  # 轨迹输入维度 (x, y, △v, 时间戳, 填充掩码)
    'hidden_dim': 128,  # 轨迹特征隐藏维度
    'encoder_depth': 4,  # Transformer编码器深度
    'num_heads': 8,  # 注意力头数
    'mlp_ratio': 4.0,  # MLP隐藏层比例
    'qkv_bias': False,  # 是否使用QKV偏置
    'drop_path': 0.2,  # 随机路径丢弃率
    'future_steps': 60,  # 预测未来60步
    'seq_len': 50,  # 输入轨迹序列长度
    'decoder_type': 'mlp',  # 解码器类型，'mlp'或'detr'
    'k': 6,  # 预测轨迹的数量
}

# 训练配置
train_config = {
    'batch_size': 512,  # 增大batch_size到512
    'learning_rate': 2e-4,  # 按线性比例调整学习率
    # 建议：batch_size=512时设置为2e-4，batch_size=1024时设置为4e-4
    'epochs': {
        'stage0': 50,  # 轨迹网络预训练
        'stage1': 30,  # 情绪和分心编码器训练
        'stage2': 30,  # FiLM调制训练
        'stage3': 50,  # 联合微调
    },
    'weight_decay': 1e-5,
    'beta': 0.1,  # VAE的KL散度权重
    'device': 'cuda',
}

# 损失函数权重配置
loss_config = {
    'lambda_traj': 1.0,  # 轨迹损失权重
    'lambda_kl_e': 1.0,  # 情绪KL散度损失权重
    'lambda_kl_d': 1.0,  # 分心KL散度损失权重
    'lambda_orth': 1.0,  # 正交损失权重
}

# 数据集配置
data_config = {
    'data_dir': 'datasets/dataset/',
    # 'train_split': 0.8,
    # 'val_split': 0.1,
    # 'test_split': 0.1,
    'shuffle': True,
    'num_samples': 10000,  # 数据集样本数
    'val_split': 0.2,  # 验证集比例
}

# 其他配置
other_config = {
    'log_dir': 'logs/',
    'checkpoint_dir': 'checkpoints/',
    'save_interval': 10,  # 每10个epoch保存一次模型
    'eval_interval': 5,  # 每5个epoch评估一次
}