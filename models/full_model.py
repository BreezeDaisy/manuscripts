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
            num_classes=config.get('distraction_num_classes', 10)  # 默认10种分心类别
        )
        
        # 初始化轨迹编码器
        self.trajectory_encoder = TrajectoryEncoder(
            input_dim=config.get('trajectory_input_dim', 5),
            hidden_dim=config.get('hidden_dim', 128),
            encoder_depth=config.get('encoder_depth', 4),
            num_heads=config.get('num_heads', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            qkv_bias=config.get('qkv_bias', False),
            drop_path=config.get('drop_path', 0.2)
        )
        
        self.film_module = FiLMModule(
            latent_dim=config['latent_dim'],
            hidden_dim=config.get('hidden_dim', 128)
        )
        
        # 初始化轨迹解码器
        self.trajectory_decoder = TrajectoryDecoder(
            hidden_dim=config.get('hidden_dim', 128),
            future_steps=config.get('future_steps', 60),
            decoder_type=config.get('decoder_type', 'mlp'),
            k=config.get('k', 6)
        )
    
    def forward(self, x_e=None, label_e=None, x_d=None, label_d=None, trajectory_data=None, use_trajectory_feature=False, return_classification=False):
        """
        前向传播
        
        Args:
            x_e: 情绪输入（驾驶员图像）[batch_size, 3, 224, 224] 或 None（使用轨迹特征时）
            label_e: 情绪类别标签 [batch_size] 或 None（使用轨迹特征时）
            x_d: 分心输入 [batch_size, 3, 224, 224] 或 None（使用轨迹特征时）
            label_d: 分心类别标签 [batch_size] 或 None（使用轨迹特征时）
            trajectory_data: 轨迹数据字典，包含轨迹和车道信息
            use_trajectory_feature: 是否使用轨迹特征作为情绪和分心编码器的输入
            return_classification: 是否返回分类预测
        
        Returns:
            pred_dict: 包含预测结果的字典
            z_e: 情绪潜在表示 [batch_size, latent_dim]
            z_d: 分心潜在表示 [batch_size, latent_dim]
            mu_e: 情绪VAE均值 [batch_size, latent_dim]
            logvar_e: 情绪VAE对数方差 [batch_size, latent_dim]
            mu_d: 分心VAE均值 [batch_size, latent_dim]
            logvar_d: 分心VAE对数方差 [batch_size, latent_dim]
            pred_e: 情绪分类预测 [batch_size, num_classes]（仅当return_classification=True时）
            pred_d: 分心分类预测 [batch_size, num_classes]（仅当return_classification=True时）
        """
        # 编码轨迹信息
        x_agent, x_encoder, key_padding_mask = self.trajectory_encoder(trajectory_data)
        
        # 编码情绪信息
        if use_trajectory_feature:
            # 使用轨迹特征作为情绪编码器的输入
            if return_classification:
                mu_e, logvar_e, z_e, pred_e = self.emotion_encoder(None, label_e, x_agent, return_classification=True)
            else:
                mu_e, logvar_e, z_e = self.emotion_encoder(None, label_e, x_agent)
        else:
            # 使用图像作为情绪编码器的输入
            if return_classification:
                mu_e, logvar_e, z_e, pred_e = self.emotion_encoder(x_e, label_e, return_classification=True)
            else:
                mu_e, logvar_e, z_e = self.emotion_encoder(x_e, label_e)
        
        # 编码分心信息
        if use_trajectory_feature:
            # 使用轨迹特征作为分心编码器的输入
            if return_classification:
                mu_d, logvar_d, z_d, pred_d = self.distraction_encoder(None, label_d, x_agent, return_classification=True)
            else:
                mu_d, logvar_d, z_d = self.distraction_encoder(None, label_d, x_agent)
        else:
            # 使用图像作为分心编码器的输入
            if return_classification:
                mu_d, logvar_d, z_d, pred_d = self.distraction_encoder(x_d, label_d, return_classification=True)
            else:
                mu_d, logvar_d, z_d = self.distraction_encoder(x_d, label_d)
        
        # FiLM调制
        x_agent_modulated = self.film_module(x_agent, z_e, z_d)
        
        # 解码预测未来轨迹
        num_agents = trajectory_data["x"].shape[1]
        pred_dict = self.trajectory_decoder(x_agent_modulated, x_encoder, key_padding_mask, num_agents)
        
        if return_classification:
            return pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_e, pred_d
        
        return pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d