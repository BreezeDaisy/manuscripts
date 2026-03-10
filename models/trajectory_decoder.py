import torch
import torch.nn as nn

class TrajectoryDecoder(nn.Module):
    """轨迹解码器 - 预测未来轨迹"""
    def __init__(self, hidden_dim, output_dim, future_steps):
        """
        初始化轨迹解码器
        
        Args:
            hidden_dim: 输入隐藏特征维度
            output_dim: 输出轨迹特征维度
            future_steps: 预测的未来步数
        """
        super(TrajectoryDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.future_steps = future_steps
        
        # 初始化LSTM的隐藏状态
        self.fc_init_h = nn.Linear(hidden_dim, hidden_dim)
        self.fc_init_c = nn.Linear(hidden_dim, hidden_dim)
        
        # LSTM解码器
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,
            bidirectional=False
        )
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, h):
        """
        前向传播
        
        Args:
            h: 编码后的轨迹特征 [batch_size, hidden_dim]
            
        Returns:
            pred: 预测的未来轨迹 [batch_size, future_steps, output_dim]
        """
        batch_size = h.size(0)
        
        # 初始化LSTM的隐藏状态和细胞状态
        h_0 = self.fc_init_h(h).unsqueeze(0).repeat(2, 1, 1)  # [2, batch_size, hidden_dim]
        c_0 = self.fc_init_c(h).unsqueeze(0).repeat(2, 1, 1)  # [2, batch_size, hidden_dim]
        
        # 初始输入（使用零向量）
        input = torch.zeros(batch_size, 1, self.output_dim, device=h.device)
        
        # 存储预测结果
        predictions = []
        
        # 自回归解码
        for _ in range(self.future_steps):
            # LSTM前向传播
            output, (h_0, c_0) = self.lstm(input, (h_0, c_0))
            
            # 预测当前步
            pred = self.fc_out(output[:, 0, :])
            predictions.append(pred)
            
            # 将当前预测作为下一步的输入
            input = pred.unsqueeze(1)
        
        # 拼接所有预测结果
        pred = torch.stack(predictions, dim=1)  # [batch_size, future_steps, output_dim]
        
        return pred