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
    def __init__(self, k=6, miss_threshold=2.0):
        """
        初始化轨迹损失
        
        Args:
            k: 预测轨迹的数量
            miss_threshold: MR评价指标的阈值
        """
        super(TrajectoryLoss, self).__init__()
        self.k = k
        self.miss_threshold = miss_threshold
    
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
        
        # 计算MR (Miss Rate)
        missed_pred = fde > self.miss_threshold
        mr = missed_pred.all(-1).float().mean()
        
        # 总损失 - 参考trainer_forecast.py的实现
        B = pred.shape[0]
        B_range = range(B)
        
        l2_norm = torch.norm(pred[..., :2] - target.unsqueeze(1), dim=-1).sum(-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = pred[B_range, best_mode]
        
        # 回归损失
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], target)
        
        # 分类损失
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach()) if pi is not None else 0
        
        # 总损失
        loss = agent_reg_loss + agent_cls_loss
        
        return loss, min_ade, min_fde, mr
    
    def compute_metrics(self, pred, target, pi=None):
        """
        计算评价指标
        
        Args:
            pred: 预测的轨迹 [batch_size, k, future_steps, 2]
            target: 真实的轨迹 [batch_size, future_steps, 5]
            pi: 预测的概率分布 [batch_size, k]
            
        Returns:
            dict: 包含minADE, minFDE, MR的字典
        """
        # 只使用目标轨迹的前2个维度（x和y坐标）
        target = target[..., :2]
        
        # 如果提供了概率分布，按概率排序
        if pi is not None:
            pred, _ = sort_predictions(pred, pi, k=self.k)
        
        # 计算minADE
        ade = torch.norm(
            pred[..., :2] - target.unsqueeze(1)[..., :2], p=2, dim=-1
        ).mean(-1)
        min_ade = ade.min(-1)[0].mean().item()
        
        # 计算minFDE
        fde = torch.norm(
            pred[..., -1, :2] - target.unsqueeze(1)[..., -1, :2], p=2, dim=-1
        )
        min_fde = fde.min(-1)[0].mean().item()
        
        # 计算MR (Miss Rate)
        missed_pred = fde > self.miss_threshold
        mr = missed_pred.all(-1).float().mean().item()
        
        return {
            'minADE': min_ade,
            'minFDE': min_fde,
            'MR': mr
        }