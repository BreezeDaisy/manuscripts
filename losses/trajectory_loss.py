import torch
import torch.nn as nn
import torch.nn.functional as F


def sort_predictions(predictions, probability, k=6):
    """Sort the predictions based on the probability of each mode.
    Args:
        predictions (torch.Tensor): The predicted trajectories [b, k, t, 2].
        probability (torch.Tensor): The probability of each mode [b, k].
    Returns:
        torch.Tensor: The sorted predictions [b, k', t, 2].
    """
    indices = torch.argsort(probability, dim=-1, descending=True)
    sorted_prob = probability[torch.arange(probability.size(0))[:, None], indices]
    sorted_predictions = predictions[
        torch.arange(predictions.size(0))[:, None], indices
    ]
    return sorted_predictions[:, :k], sorted_prob[:, :k]


class TrajectoryLoss(nn.Module):
    """轨迹损失 - 使用minADE和minFDE"""
    def __init__(self, k=6):
        """
        初始化轨迹损失
        
        Args:
            k: 预测轨迹的数量
        """
        super(TrajectoryLoss, self).__init__()
        self.k = k
    
    def forward(self, pred, target, pi=None):
        """
        计算轨迹损失
        
        Args:
            pred: 预测的轨迹 [batch_size, k, future_steps, 2]
            target: 真实的轨迹 [batch_size, future_steps, 5] 或 None
            pi: 预测的概率分布 [batch_size, k]
            
        Returns:
            loss: 轨迹损失，如果target为None则返回0
        """
        if target is None:
            return 0.0
        
        # 只使用目标轨迹的前2个维度（x和y坐标）
        target = target[..., :2]
        
        # 如果提供了概率分布，按概率排序
        if pi is not None:
            pred, _ = sort_predictions(pred, pi, k=self.k)
        
        # 计算minADE
        ade = torch.norm(
            pred[..., :2] - target.unsqueeze(1)[..., :2], p=2, dim=-1
        ).mean(-1)
        min_ade = ade.min(-1)[0].mean()
        
        # 计算minFDE
        fde = torch.norm(
            pred[..., -1, :2] - target.unsqueeze(1)[..., -1, :2], p=2, dim=-1
        )
        min_fde = fde.min(-1)[0].mean()
        
        # 总损失
        loss = min_ade + min_fde
        
        return loss