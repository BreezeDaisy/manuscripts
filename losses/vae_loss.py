import torch
import torch.nn as nn


class VAELoss(nn.Module):
    """VAE损失 - KL散度"""
    def __init__(self, beta=1.0):
        """
        初始化VAE损失
        
        Args:
            beta: KL散度的权重
        """
        super(VAELoss, self).__init__()
        self.beta = beta
    
    def forward(self, mu, logvar):
        """
        计算KL散度损失
        
        KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        
        Args:
            mu: 均值 [batch_size, latent_dim]
            logvar: 对数方差 [batch_size, latent_dim]
            
        Returns:
            kl_loss: KL散度损失
        """
        # 计算KL散度
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 乘以权重
        kl_loss = self.beta * kl_loss
        
        return kl_loss