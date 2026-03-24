import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryDecoder(nn.Module):
    """轨迹解码器 - 预测未来轨迹"""
    def __init__(self, hidden_dim=128, future_steps=60, decoder_type="mlp", k=6):
        """
        初始化轨迹解码器
        
        Args:
            hidden_dim: 输入隐藏特征维度
            future_steps: 预测的未来步数
            decoder_type: 解码器类型，"mlp"或"detr"
            k: 预测轨迹的数量
        """
        super(TrajectoryDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps
        self.decoder_type = decoder_type
        self.k = k
        
        # 加载对应的解码器
        if decoder_type == "mlp":
            from .layers.multimodal_decoder_emp import MultimodalDecoder
        elif decoder_type == "detr":
            from .layers.multimodal_decoder_emp_attn import MultimodalDecoder
        else:
            raise ValueError(f"Unknown Decoder Type: {decoder_type} (must be 'mlp' or 'detr')")
        
        # 解码器
        self.decoder = MultimodalDecoder(hidden_dim, future_steps, k=k)
        
        # 其他车辆轨迹预测器
        self.dense_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, future_steps * 2)
        )
    
    def forward(self, x_agent, x_encoder, key_padding_mask, num_agents):
        """
        前向传播
        
        Args:
            x_agent: 目标车辆的编码特征 [batch_size, hidden_dim]
            x_encoder: 编码后的场景特征 [batch_size, num_agents+num_lanes, hidden_dim]
            key_padding_mask: 场景关键填充掩码 [batch_size, num_agents+num_lanes]
            num_agents: 车辆数量
        
        Returns:
            包含预测结果的字典:
                - y_hat: 预测的目标车辆未来轨迹 [batch_size, k, future_steps, 2]
                - pi: 预测的概率分布 [batch_size, k]
                - y_hat_others: 其他车辆的预测轨迹 [batch_size, num_agents-1, future_steps, 2]
                - y_hat_eps: 预测的轨迹终点 [batch_size, k, 2]
                - x_agent: 目标车辆的编码特征 [batch_size, hidden_dim]
        """
        # 预测其他车辆轨迹
        x_others = x_encoder[:, 1:num_agents]
        y_hat_others = self.dense_predictor(x_others).view(x_others.shape[0], -1, self.future_steps, 2)
        
        # 预测目标车辆轨迹
        y_hat, pi = self.decoder(x_agent, x_encoder, key_padding_mask, num_agents)
        
        # 提取轨迹终点
        y_hat_eps = y_hat[:, :, -1]
        
        return {
            "y_hat": y_hat,
            "pi": pi,
            "y_hat_others": y_hat_others,
            "y_hat_eps": y_hat_eps,
            "x_agent": x_agent
        }