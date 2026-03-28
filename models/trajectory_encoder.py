import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block

class TrajectoryEncoder(nn.Module):
    """轨迹编码器 - 使用Transformer处理历史轨迹"""
    def __init__(self, input_dim=5, hidden_dim=128, encoder_depth=4, num_heads=8, mlp_ratio=4.0, qkv_bias=True, drop_path=0.2):
        """
        初始化轨迹编码器
        
        Args:
            input_dim: 输入轨迹特征维度
            hidden_dim: 隐藏层维度
            encoder_depth: Transformer编码器深度
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层比例
            qkv_bias: 是否使用QKV偏置
            drop_path: 随机路径丢弃率
        """
        super(TrajectoryEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_depth = encoder_depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_path = drop_path
        
        self.history_steps = 50
        self.future_steps = 60
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        
        # 历史轨迹投影
        self.h_proj = nn.Linear(5, hidden_dim)
        
        # 历史轨迹嵌入层
        self.h_embed = nn.ModuleList(
            Block(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                cross_attn=False
            )
            for i in range(encoder_depth)
        )
        
        # 车道嵌入层
        self.lane_embed = LaneEmbeddingLayer(3, hidden_dim)
        
        # 位置嵌入
        self.pos_embed = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList(
            Block(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                cross_attn=False
            )
            for i in range(encoder_depth)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 类型嵌入
        self.actor_type_embed = nn.Parameter(torch.Tensor(4, hidden_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, hidden_dim))
        
        # 初始化权重
        self.initialize_weights()
    
    def initialize_weights(self):
        """初始化权重"""
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化权重的辅助函数"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, data):
        """
        前向传播
        
        Args:
            data: 包含轨迹和车道信息的字典
                - x: 车辆历史轨迹的位置数据 [batch_size, num_agents, history_steps, 2]
                - x_padding_mask: 轨迹填充掩码 [batch_size, num_agents, history_steps]
                - x_key_padding_mask: 轨迹关键填充掩码 [batch_size, num_agents]
                - x_velocity_diff: 速度差异 [batch_size, num_agents, history_steps]
                - x_centers: 轨迹中心点 [batch_size, num_agents, 2]
                - x_angles: 轨迹角度 [batch_size, num_agents, history_steps+1]
                - x_attr: 轨迹属性 [batch_size, num_agents, attr_dim]
                - lane_positions: 车道位置 [batch_size, num_lanes, lane_points, 2]
                - lane_centers: 车道中心点 [batch_size, num_lanes, 2]
                - lane_angles: 车道角度 [batch_size, num_lanes]
                - lane_padding_mask: 车道填充掩码 [batch_size, num_lanes, lane_points]
                - lane_key_padding_mask: 车道关键填充掩码 [batch_size, num_lanes]
        
        Returns:
            x_agent: 目标车辆的编码特征 [batch_size, hidden_dim]
            x_encoder: 编码后的场景特征 [batch_size, num_agents+num_lanes, hidden_dim]
            key_padding_mask: 场景关键填充掩码 [batch_size, num_agents+num_lanes]
        """
        hist_padding_mask = data["x_padding_mask"][:, :, :self.history_steps]
        hist_key_padding_mask = data["x_key_padding_mask"]
        
        # 构建历史轨迹特征
        # 只使用x的前2个维度（x, y坐标）
        x_pos = data["x"][..., :2]
        
        hist_feat = torch.cat(
            [
                x_pos,
                data["x_velocity_diff"][..., None],
                (~hist_padding_mask[..., None]).float(),
            ],
            dim=-1,
        )
        
        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)
        
        # AGENT ENCODING
        actor_feat = hist_feat[~hist_feat_key_padding]
        
        # 添加时间戳
        ts = torch.arange(self.history_steps).view(1, -1, 1).repeat(actor_feat.shape[0], 1, 1).to(actor_feat.device).float()
        actor_feat = torch.cat([actor_feat, ts], dim=-1)
        
        # 投影和嵌入
        actor_feat = self.h_proj(actor_feat)
        kpm = hist_padding_mask.view(B*N, -1)[~hist_feat_key_padding]
        for blk in self.h_embed:
            actor_feat = blk(actor_feat, key_padding_mask=kpm)
        
        # 最大池化
        actor_feat = torch.max(actor_feat, axis=1).values
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device, dtype=actor_feat.dtype
        )
        
        # 填充回原始形状
        mask_indices = torch.nonzero(~hist_feat_key_padding, as_tuple=True)
        actor_feat_tmp.scatter_(0, mask_indices[0].unsqueeze(1).expand(-1, self.hidden_dim), actor_feat)
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])
        
        # LANE ENCODING
        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, (~lane_padding_mask[..., None]).float()], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)
        
        # POS ENCODING
        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, self.history_steps-1], data["lane_angles"]], dim=1)    
        
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)      
        pos_embed = self.pos_embed(pos_feat)
        
        # SCENE ENCODING
        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed
        
        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat([data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)           
        
        x_encoder = x_encoder + pos_embed
        
        # Transformer编码
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)
        
        # 提取目标车辆特征
        x_agent = x_encoder[:, 0] 
        
        return x_agent, x_encoder, key_padding_mask