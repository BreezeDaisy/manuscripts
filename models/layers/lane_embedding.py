import torch
import torch.nn as nn

class LaneEmbeddingLayer(nn.Module):
    """车道嵌入层"""
    def __init__(self, input_dim, embed_dim):
        """
        初始化车道嵌入层
        
        Args:
            input_dim: 输入维度
            embed_dim: 嵌入维度
        """
        super(LaneEmbeddingLayer, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # 车道点嵌入
        self.lane_point_embed = nn.Linear(input_dim, embed_dim)
        
        # 车道聚合
        self.lane_aggregation = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 车道点数据 [batch_size * num_lanes, num_points, input_dim]
        
        Returns:
            lane_emb: 车道嵌入 [batch_size * num_lanes, embed_dim]
        """
        # 嵌入车道点
        x = self.lane_point_embed(x)
        
        # 聚合车道点特征（最大池化）
        x = torch.max(x, dim=1)[0]
        
        # 进一步处理车道特征
        lane_emb = self.lane_aggregation(x)
        
        return lane_emb