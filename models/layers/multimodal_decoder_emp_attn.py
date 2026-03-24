import torch
import torch.nn as nn
from .transformer_blocks import Block

class MultimodalDecoder(nn.Module):
    def __init__(self, embed_dim, future_steps, k=6):
        super().__init__()
        self.k = k
        self.future_steps = future_steps
        self.embed_dim = embed_dim

        self.query_embed = nn.Embedding(k, embed_dim)
        self.decoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads=8, mlp_ratio=4.0, qkv_bias=False, drop_path=0.0, cross_attn=True)
            for _ in range(2)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, future_steps * 2)
        )

        self.pi = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_agent, x_encoder, key_padding_mask, num_agents):
        batch_size = x_agent.shape[0]

        # 创建查询嵌入
        query_embed = self.query_embed(torch.arange(self.k, device=x_agent.device))
        query_embed = query_embed.unsqueeze(0).repeat(batch_size, 1, 1)

        # 添加目标车辆特征到查询
        x_agent_expanded = x_agent.unsqueeze(1).repeat(1, self.k, 1)
        query = query_embed + x_agent_expanded

        # 解码
        for blk in self.decoder_blocks:
            query = blk(query, key_padding_mask=key_padding_mask, cross=x_encoder)

        # 预测轨迹
        pred = self.decoder(query)
        pred = pred.view(batch_size, self.k, self.future_steps, 2)

        # 预测概率
        pi = self.pi(query).squeeze(-1)
        pi = torch.softmax(pi, dim=-1)

        return pred, pi