import torch
import torch.nn as nn

class TrajectoryEncoder(nn.Module):
    """轨迹编码器 - 使用LSTM处理历史轨迹"""
    def __init__(self, input_dim, hidden_dim):
        """
        初始化轨迹编码器
        
        Args:
            input_dim: 输入轨迹特征维度
            hidden_dim: 隐藏层维度
        """
        super(TrajectoryEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 使用LSTM处理时序轨迹数据
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,
            bidirectional=False
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入轨迹序列 [batch_size, seq_len, input_dim]
            
        Returns:
            h: 编码后的轨迹特征 [batch_size, hidden_dim]
        """
        # LSTM前向传播
        # output: [batch_size, seq_len, hidden_dim]
        # h_n: [num_layers, batch_size, hidden_dim]
        # c_n: [num_layers, batch_size, hidden_dim]
        output, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一层的隐藏状态作为轨迹特征
        h = h_n[-1]  # [batch_size, hidden_dim]
        
        return h