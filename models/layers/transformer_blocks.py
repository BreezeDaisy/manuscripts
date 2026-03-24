import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """Transformer块"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop_path=0.0, cross_attn=False):
        """
        初始化Transformer块
        
        Args:
            dim: 输入维度
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层比例
            qkv_bias: 是否使用QKV偏置
            drop_path: 随机路径丢弃率
            cross_attn: 是否使用交叉注意力
        """
        super(Block, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_path = drop_path
        self.cross_attn = cross_attn
        
        # 自注意力层
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, batch_first=True)
        self.drop_path_layer = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # MLP层
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x, key_padding_mask=None, cross=None):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            key_padding_mask: 注意力掩码 [batch_size, seq_len]
            cross: 交叉注意力的键值对 [batch_size, cross_seq_len, dim]
        
        Returns:
            x: 输出特征 [batch_size, seq_len, dim]
        """
        # 自注意力
        x_norm = self.norm1(x)
        if self.cross_attn and cross is not None:
            cross_norm = self.norm1(cross)
            attn_output, _ = self.attn(x_norm, cross_norm, cross_norm, key_padding_mask=key_padding_mask)
        else:
            attn_output, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        
        # 残差连接
        x = x + self.drop_path_layer(attn_output)
        
        # MLP
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        
        # 残差连接
        x = x + self.drop_path_layer(mlp_output)
        
        return x