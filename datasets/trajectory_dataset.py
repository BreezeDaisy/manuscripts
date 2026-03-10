import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """轨迹数据集"""
    def __init__(self, num_samples=1000, seq_len=10, future_steps=30, input_dim=4):
        """
        初始化轨迹数据集
        
        Args:
            num_samples: 样本数量
            seq_len: 历史轨迹序列长度
            future_steps: 未来轨迹预测步数
            input_dim: 轨迹特征维度
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.input_dim = input_dim
    
    def __len__(self):
        """
        返回数据集长度
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            history: 历史轨迹 [seq_len, input_dim]
            future: 未来轨迹 [future_steps, input_dim]
        """
        # 生成随机历史轨迹
        history = torch.randn(self.seq_len, self.input_dim)
        
        # 生成随机未来轨迹
        future = torch.randn(self.future_steps, self.input_dim)
        
        return history, future