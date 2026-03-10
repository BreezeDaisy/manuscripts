# 导入必要的库
import torch

# 模型配置
model_config = {
    # 情绪编码器配置
    'latent_dim': 32,  # VAE潜在空间维度
    'emotion_num_classes': 7,  # 情绪类别数量
    'distraction_num_classes': 5,  # 分心类别数量
    
    # 轨迹编码器配置
    'trajectory_input_dim': 4,  # 轨迹输入维度 (x, y, vx, vy)
    'hidden_dim': 256,  # 轨迹特征隐藏维度
    'trajectory_output_dim': 4,  # 轨迹输出维度
    'future_steps': 30,  # 预测未来30步
    'seq_len': 10,  # 输入轨迹序列长度，大概1s
}

# 训练配置
train_config = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': {
        'stage0': 50,  # 轨迹网络预训练
        'stage1': 30,  # 情绪和分心编码器训练
        'stage2': 30,  # FiLM调制训练
        'stage3': 50,  # 联合微调
    },
    'weight_decay': 1e-5,
    'beta': 0.1,  # VAE的KL散度权重
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
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
    'data_dir': 'datasets/',
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'shuffle': True,
    'num_samples': 1000,  # 示例数据集样本数
}

# 其他配置
other_config = {
    'log_dir': 'logs/',
    'checkpoint_dir': 'checkpoints/',
    'save_interval': 10,  # 每10个epoch保存一次模型
    'eval_interval': 5,  # 每5个epoch评估一次
}