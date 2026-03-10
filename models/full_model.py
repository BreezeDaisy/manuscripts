import torch
import torch.nn as nn
from .emotion_encoder import EmotionEncoder
from .distraction_encoder import DistractionEncoder
from .trajectory_encoder import TrajectoryEncoder
from .trajectory_decoder import TrajectoryDecoder
from .film_module import FiLMModule

class DrivingBehaviorModel(nn.Module):
    """完整的驾驶行为调制模型"""
    def __init__(self, config):
        """
        初始化完整模型
        
        Args:
            config: 模型配置参数
        """
        super(DrivingBehaviorModel, self).__init__()
        
        # 初始化各组件
        self.emotion_encoder = EmotionEncoder(
            latent_dim=config['latent_dim'],
            num_classes=config.get('emotion_num_classes', 7)  # 默认7种情绪类别
        )
        
        self.distraction_encoder = DistractionEncoder(
            latent_dim=config['latent_dim'],
            num_classes=config.get('distraction_num_classes', 5)  # 默认5种分心类别
        )
        
        self.trajectory_encoder = TrajectoryEncoder(
            input_dim=config['trajectory_input_dim'],
            hidden_dim=config['hidden_dim']
        )
        
        self.film_module = FiLMModule(
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim']
        )
        
        self.trajectory_decoder = TrajectoryDecoder(
            hidden_dim=config['hidden_dim'],
            output_dim=config['trajectory_output_dim'],
            future_steps=config['future_steps']
        )
    
    def forward(self, x_e, label_e, x_d, label_d, x_t):
        """
        前向传播
        
        Args:
            x_e: 情绪输入（驾驶员图像）[batch_size, 3, 224, 224]
            label_e: 情绪类别标签 [batch_size]
            x_d: 分心输入 [batch_size, 3, 224, 224]
            label_d: 分心类别标签 [batch_size]
            x_t: 轨迹输入 [batch_size, seq_len, trajectory_input_dim]
            
        Returns:
            pred: 预测的未来轨迹 [batch_size, future_steps, trajectory_output_dim]
            z_e: 情绪潜在表示 [batch_size, latent_dim]
            z_d: 分心潜在表示 [batch_size, latent_dim]
            mu_e: 情绪VAE均值 [batch_size, latent_dim]
            logvar_e: 情绪VAE对数方差 [batch_size, latent_dim]
            mu_d: 分心VAE均值 [batch_size, latent_dim]
            logvar_d: 分心VAE对数方差 [batch_size, latent_dim]
        """
        # 编码情绪信息
        mu_e, logvar_e, z_e = self.emotion_encoder(x_e, label_e)
        
        # 编码分心信息
        mu_d, logvar_d, z_d = self.distraction_encoder(x_d, label_d)
        
        # 编码轨迹信息
        h = self.trajectory_encoder(x_t)
        
        # FiLM调制
        h_modulated = self.film_module(h, z_e, z_d)
        
        # 解码预测未来轨迹
        pred = self.trajectory_decoder(h_modulated)
        
        return pred, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d