import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# 添加当前文件所在目录的父目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.reparameterize import reparameterize

class ChannelAttention(nn.Module):
    """
    通道注意力模块
    使用全局平均池化和最大池化来捕获通道维度上的信息
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch_size, seq_len, in_channels]
        batch_size, seq_len, in_channels = x.size()
        
        # 平均池化路径
        avg_out = self.avg_pool(x.transpose(1, 2)).squeeze(-1)  # [batch_size, in_channels]
        avg_out = self.fc(avg_out).unsqueeze(2)  # [batch_size, in_channels, 1]
        
        # 最大池化路径
        max_out = self.max_pool(x.transpose(1, 2)).squeeze(-1)  # [batch_size, in_channels]
        max_out = self.fc(max_out).unsqueeze(2)  # [batch_size, in_channels, 1]
        
        # 拼接两条路径的输出
        out = avg_out + max_out
        out = out.transpose(1, 2)  # [batch_size, 1, in_channels]
        
        # 应用注意力权重
        return x * out.expand_as(x)

class SpatialAttention(nn.Module):
    """
    空间注意力模块（基于稀疏自注意力机制）
    采用局部窗口+稀疏采样策略，在保持长距离依赖建模能力的同时提高参数效率
    """
    def __init__(self, in_channels, kernel_size=7, window_size=16, num_global_tokens=4, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        
        # 轻量级投影层
        self.query_proj = nn.Linear(in_channels, in_channels)
        self.key_proj = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
        
        # 残差卷积路径（更轻量级）
        padding = kernel_size // 2
        self.residual_conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
        # 全局token投影（用于稀疏全局连接）
        self.global_token_proj = nn.Linear(in_channels, in_channels // 4)  # 降维以提高效率
        self.global_out_proj = nn.Linear(in_channels // 4, in_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [batch_size, seq_len, in_channels]
        batch_size, seq_len, in_channels = x.size()
        
        # 1. 局部窗口注意力（高效的局部依赖建模）
        # 计算查询、键、值
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        
        # 初始化输出
        x_local_attn = torch.zeros_like(x)
        
        # 分窗口处理注意力
        for i in range(0, seq_len, self.window_size):
            # 获取当前窗口
            window_end = min(i + self.window_size, seq_len)
            window_len = window_end - i
            
            # 提取窗口内的Q、K、V
            q_window = q[:, i:window_end]
            k_window = k[:, i:window_end]
            v_window = v[:, i:window_end]
            
            # 计算缩放点积注意力（仅在窗口内）
            scale = 1.0 / math.sqrt(in_channels)
            attn = torch.matmul(q_window, k_window.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            
            # 应用注意力
            window_out = torch.matmul(attn, v_window)
            x_local_attn[:, i:window_end] = window_out
        
        # 2. 稀疏全局连接（捕获长距离依赖）
        if seq_len > self.window_size and self.num_global_tokens > 0:
            # 均匀采样全局token的索引
            global_indices = torch.linspace(0, seq_len - 1, self.num_global_tokens, dtype=torch.long, device=x.device)
            
            # 提取全局token
            global_tokens = x[:, global_indices] # 选中global_indices索引对应的token
            
            # 为每个token计算与全局token的注意力
            # 先对query进行降维，确保维度匹配
            q_global = self.global_token_proj(q) # 对q进行降维,将in_channels映射到in_channels//4
            k_global = self.global_token_proj(global_tokens)
            v_global = self.global_token_proj(global_tokens)
            
            # 计算全局注意力
            scale = 1.0 / math.sqrt(in_channels // 4)  # 基于降维后的维度缩放
            global_attn = torch.matmul(q_global, k_global.transpose(-2, -1)) * scale
            global_attn = global_attn.softmax(dim=-1)
            global_attn = self.dropout(global_attn)
            
            # 应用全局注意力
            x_global_attn = torch.matmul(global_attn, v_global)
            x_global_attn = self.global_out_proj(x_global_attn)  # 升维
        else:
            x_global_attn = torch.zeros_like(x)
        
        # 3. 轻量级残差卷积路径
        avg_out = torch.mean(x, dim=2, keepdim=True)  # [batch_size, seq_len, 1]
        max_out, _ = torch.max(x, dim=2, keepdim=True)  # [batch_size, seq_len, 1]
        out_conv = torch.cat([avg_out, max_out], dim=2)  # [batch_size, seq_len, 2]
        out_conv = out_conv.transpose(1, 2)  # [batch_size, 2, seq_len]
        out_conv = self.residual_conv(out_conv).transpose(1, 2)  # [batch_size, seq_len, 1]
        out_conv = self.sigmoid(out_conv).expand_as(x)  # [batch_size, seq_len, in_channels]
        
        # 4. 融合所有路径
        out = x + x_local_attn + x_global_attn  # 残差连接
        out = out * out_conv  # 应用卷积注意力权重
        out = self.out_proj(out)  # 最终投影
        
        return out

class AttentionModule(nn.Module):
    """
    组合通道注意力和空间注意力的模块
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels, kernel_size)
    
    def forward(self, x):
        # 先应用通道注意力，再应用空间注意力
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class MambaBlock(nn.Module):
    """
    Mamba块,包含Mamba层、归一化、注意力机制和残差连接
    """
    def __init__(self, dim, ssm_rank=64, dropout_rate=0.1, use_attention=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # 简化版Mamba块，使用Conv1d替代真实的Mamba实现
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_attention = use_attention
        if use_attention:
            # 添加注意力模块
            self.attention = AttentionModule(dim)
    
    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)  # [batch_size, dim, seq_len]
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # [batch_size, seq_len, dim]
        
        # 应用注意力机制
        if self.use_attention:
            x = self.attention(x)
            
        x = self.dropout(x)
        x = x + residual
        return x

class MambaStage(nn.Module):
    """
    Mamba阶段,包含多个Mamba块
    """
    def __init__(self, dim, depth, ssm_rank=64, dropout_rate=0.1, use_attention=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(dim, ssm_rank, dropout_rate, use_attention) for _ in range(depth)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

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
        
        # 初始卷积特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 中间卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 计算序列长度（用于Mamba）
        self.seq_len = (224 // 16) ** 2
        
        # 三阶段Mamba
        self.mamba_stages = nn.ModuleList()
        current_dim = 256
        
        for depth in [2, 2, 2]:
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(MambaBlock(current_dim, 64, 0.1, True))
            self.mamba_stages.append(nn.Sequential(*stage_blocks))
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 标签嵌入层 - 设置为更大的大小以容纳实际的标签值范围
        self.label_embedding = nn.Embedding(20, latent_dim)
        
        # 轨迹特征投影层 - 将128维轨迹特征映射到256维，与卷积特征维度匹配
        self.trajectory_proj = nn.Linear(128, 256)
        
        # 全连接层映射到均值和对数方差（输入维度包含标签嵌入）
        self.fc_mu = nn.Linear(256 + latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(256 + latent_dim, latent_dim)
        
        # 分类头部
        self.classifier = nn.Linear(256, self.num_classes)
        
    def forward(self, x=None, label=None, trajectory_feature=None, return_classification=False):
        """
        前向传播
        
        Args:
            x: 输入分心相关图像 [batch_size, 3, 224, 224] 或 None（使用轨迹特征时）
            label: 分心类别标签 [batch_size] 或 None（使用轨迹特征时）
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
            # 通过投影层将128维轨迹特征映射到256维
            h = self.trajectory_proj(trajectory_feature)
            # 当使用轨迹特征时，不需要标签
            if label is None:
                # 如果没有标签，使用全零向量作为嵌入
                batch_size = trajectory_feature.shape[0]
                label_emb = torch.zeros(batch_size, self.latent_dim, device=trajectory_feature.device)
            else:
                # 嵌入标签
                label_emb = self.label_embedding(label)
        else:
            # 通过卷积层提取特征
            x = self.conv1(x)  # [batch_size, 64, 56, 56]
            x = self.conv2(x)  # [batch_size, 128, 28, 28]
            x = self.conv3(x)  # [batch_size, 256, 14, 14]
            
            # 转换为序列格式 [batch_size, seq_len, embed_dim]
            batch_size = x.shape[0]
            x = x.flatten(2)  # [batch_size, 256, 196]
            x = x.transpose(1, 2)  # [batch_size, 196, 256]
            
            # 通过三阶段Mamba
            for stage in self.mamba_stages:
                x = stage(x)
            
            # 全局平均池化
            x = x.transpose(1, 2)  # [batch_size, 256, 196]
            h = self.global_pool(x).squeeze(-1)  # [batch_size, 256]
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