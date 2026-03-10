import torch
import torch.nn as nn


class OrthogonalLoss(nn.Module):
    """正交损失 - 避免Emotion latent和Distraction latent信息耦合"""
    def __init__(self):
        """
        初始化正交损失
        """
        super(OrthogonalLoss, self).__init__()
    
    def forward(self, z_e, z_d):
        """
        计算正交损失
        
        L_orth = || z_e^T z_d ||
        
        Args:
            z_e: 情绪潜在表示 [batch_size, latent_dim]
            z_d: 分心潜在表示 [batch_size, latent_dim]
            
        Returns:
            orth_loss: 正交损失
        """
        # 计算z_e和z_d的点积
        dot_product = torch.sum(z_e * z_d, dim=1)
        
        # 计算绝对值并求和
        orth_loss = torch.sum(torch.abs(dot_product))
        
        return orth_loss