import torch
import torch.nn as nn

class FiLMModule(nn.Module):
    """FiLM调制模块 - 根据情绪和分心潜在表示调制轨迹特征"""
    def __init__(self, latent_dim, hidden_dim):
        """
        初始化FiLM模块
        
        Args:
            latent_dim: 潜在空间维度（情绪和分心编码器的输出维度）
            hidden_dim: 轨迹特征的隐藏维度
        """
        super(FiLMModule, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 融合情绪和分心的潜在表示
        self.fc_fusion = nn.Linear(latent_dim * 2, latent_dim)
        
        # 生成调制参数α和β
        self.fc_alpha = nn.Linear(latent_dim, hidden_dim)
        self.fc_beta = nn.Linear(latent_dim, hidden_dim)
        
    def forward(self, h, z_e, z_d):
        """
        前向传播
        
        Args:
            h: 轨迹特征 [batch_size, hidden_dim]
            z_e: 情绪潜在表示 [batch_size, latent_dim]
            z_d: 分心潜在表示 [batch_size, latent_dim]
            
        Returns:
            h_modulated: 调制后的轨迹特征 [batch_size, hidden_dim]
        """
        # 融合情绪和分心的潜在表示
        z_combined = torch.cat([z_e, z_d], dim=1)  # [batch_size, latent_dim * 2]
        z_fused = self.fc_fusion(z_combined)  # [batch_size, latent_dim]
        
        # 生成调制参数α和β
        alpha = self.fc_alpha(z_fused)  # [batch_size, hidden_dim]
        beta = self.fc_beta(z_fused)  # [batch_size, hidden_dim]
        
        # 执行FiLM调制: h' = α * h + β
        h_modulated = alpha * h + beta  # [batch_size, hidden_dim]
        
        return h_modulated