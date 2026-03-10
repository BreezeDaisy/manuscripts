import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryLoss(nn.Module):
    """轨迹损失 - MSE损失"""
    def __init__(self):
        """
        初始化轨迹损失
        """
        super(TrajectoryLoss, self).__init__()
    
    def forward(self, pred, target):
        """
        计算MSE轨迹损失
        
        Args:
            pred: 预测的轨迹 [batch_size, future_steps, output_dim]
            target: 真实的轨迹 [batch_size, future_steps, output_dim]
            
        Returns:
            loss: MSE轨迹损失
        """
        # 计算均方误差
        loss = F.mse_loss(pred, target)
        
        return loss