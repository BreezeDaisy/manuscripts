import torch


def reparameterize(mu, logvar):
    """
    VAE重参数化函数
    
    公式：z = mu + sigma * epsilon，其中 epsilon ~ N(0, I)
    
    Args:
        mu: 均值 [batch_size, latent_dim]
        logvar: 对数方差 [batch_size, latent_dim]
        
    Returns:
        z: 采样的潜在向量 [batch_size, latent_dim]
    """
    # 计算标准差
    std = torch.exp(0.5 * logvar)
    
    # 从标准正态分布中采样epsilon
    eps = torch.randn_like(std)
    
    # 重参数化技巧
    z = mu + eps * std
    
    return z