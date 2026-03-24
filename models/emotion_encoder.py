import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加当前文件所在目录的父目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.reparameterize import reparameterize

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnhancedBlock(nn.Module):
    """
    增强型网络块
    结构：下采样 → 归一化 → SE激励层 → 卷积FFN层
    归一化层接受下采样后和下采样前的相加作为输入
    最后输出为归一化之前和卷积FFN后相加的结果
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # 下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      stride=stride, padding=1, bias=False),
        )
        
        # 归一化
        self.bn = nn.BatchNorm2d(out_channels)
        
        # SE激励层
        self.se = SEBlock(out_channels)
        
        # 卷积FFN层
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 跳跃连接（处理通道数和步长不同的情况）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
            )

    def forward(self, x):
        # 下采样
        downsampled = self.downsample(x)
        
        # 跳跃连接
        shortcut = self.shortcut(x)
        
        # 归一化层接受下采样后和下采样前的相加作为输入
        norm_input = downsampled + shortcut
        normed = self.bn(norm_input)
        
        # SE激励层
        se_out = self.se(normed)
        
        # 卷积FFN层
        ffn_out = self.ffn(se_out)
        
        # 最后输出为归一化之前和卷积FFN后相加的结果
        out = norm_input + ffn_out
        out = F.relu(out, inplace=True)
        
        return out

class EmotionEncoder(nn.Module):
    """情绪编码器 - 使用Conditional VAE结构处理驾驶员图像"""
    def __init__(self, latent_dim, num_classes):
        """
        初始化情绪编码器
        
        Args:
            latent_dim: 潜在空间维度
            num_classes: 情绪类别数量
        """
        super(EmotionEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # CNN编码器网络
        self.encoder = nn.Sequential(
            # 初始卷积: (3, 224, 224) → (32, 112, 112)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=False),

            # 增强型块1: (32, 112, 112) → (64, 56, 56)
            EnhancedBlock(in_channels=32, out_channels=64, stride=2),

            # 池化: (64, 56, 56) → (64, 19, 19)
            nn.MaxPool2d(kernel_size=4, stride=3, padding=1),

            # 增强型块2: (64, 19, 19) → (128, 10, 10)
            EnhancedBlock(in_channels=64, out_channels=128, stride=2),

            # 池化: (128, 10, 10) → (128, 5, 5)
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 最终池化和扁平化
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 标签嵌入层
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # 全连接层映射到均值和对数方差（输入维度包含标签嵌入）
        # 注意：当使用轨迹特征时，输入维度会根据轨迹特征的维度自动调整
        self.fc_mu = nn.Linear(128 + latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(128 + latent_dim, latent_dim)
        
        # 分类头部
        self.classifier = nn.Linear(128, self.num_classes)
        
    def forward(self, x=None, label=None, trajectory_feature=None, return_classification=False):
        """
        前向传播
        
        Args:
            x: 输入驾驶员图像 [batch_size, 3, 224, 224] 或 None（使用轨迹特征时）
            label: 情绪类别标签 [batch_size] 或 None（使用轨迹特征时）
            trajectory_feature: 轨迹特征 [batch_size, hidden_dim] 或 None（使用图像时）
            return_classification: 是否返回分类预测
            
        Returns:
            mu: 均值 [batch_size, latent_dim]
            logvar: 对数方差 [batch_size, latent_dim]
            z: 采样的潜在向量 [batch_size, latent_dim]
            pred: 分类预测 [batch_size, num_classes]（仅当return_classification=True时）
        """
        if trajectory_feature is not None:
            # 在中间层使用轨迹特征作为输入
            # 轨迹特征维度为128，与CNN编码器输出维度匹配
            h = trajectory_feature
            # 当使用轨迹特征时，不需要标签
            if label is None:
                # 如果没有标签，使用全零向量作为嵌入
                batch_size = trajectory_feature.shape[0]
                label_emb = torch.zeros(batch_size, self.latent_dim, device=trajectory_feature.device)
            else:
                # 嵌入标签
                label_emb = self.label_embedding(label)
        else:
            # 编码图像
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
        
        # 计算分类预测
        if return_classification:
            pred = self.classifier(h)
            return mu, logvar, z, pred
        
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