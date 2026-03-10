import torch
import torch.nn as nn
from ..utils.reparameterize import reparameterize

class DistractionEncoder(nn.Module):
    """分心编码器 - 使用Conditional VAE结构处理分心相关输入"""
    def __init__(self, latent_dim, num_classes):
        """
        初始化分心编码器
        
        Args:
            latent_dim: 潜在空间维度
            num_classes: 分心类别数量
        """
        super(DistractionEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # CNN编码器网络（与情绪编码器类似）
        self.encoder = nn.Sequential(
            # 输入: [batch_size, 3, 224, 224]
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 标签嵌入层
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # 全连接层映射到均值和对数方差（输入维度包含标签嵌入）
        self.fc_mu = nn.Linear(256 + latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(256 + latent_dim, latent_dim)
        
    def forward(self, x, label):
        """
        前向传播
        
        Args:
            x: 输入分心相关图像 [batch_size, 3, 224, 224]
            label: 分心类别标签 [batch_size]
            
        Returns:
            mu: 均值 [batch_size, latent_dim]
            logvar: 对数方差 [batch_size, latent_dim]
            z: 采样的潜在向量 [batch_size, latent_dim]
        """
        # 编码输入
        h = self.encoder(x)
        
        # 嵌入标签
        label_emb = self.label_embedding(label)
        
        # 拼接特征和标签嵌入
        h_combined = torch.cat([h, label_emb], dim=1)
        
        # 计算均值和对数方差
        mu = self.fc_mu(h_combined)
        logvar = self.fc_logvar(h_combined)
        
        # 采样
        z = self.sample(mu, logvar)
        
        return mu, logvar, z
    
    def sample(self, mu, logvar):
        """
        从潜在分布中采样（VAE reparameterization）
        
        Args:
            mu: 均值 [batch_size, latent_dim]
            logvar: 对数方差 [batch_size, latent_dim]
            
        Returns:
            z: 采样的潜在向量 [batch_size, latent_dim]
        """
        return reparameterize(mu, logvar)