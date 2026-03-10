import torch
from torch.utils.data import Dataset, DataLoader
from .emotion_dataset import EmotionDataset
from .distraction_dataset import DistractionDataset
from .trajectory_dataset import TrajectoryDataset

class DrivingBehaviorDataset(Dataset):
    """驾驶行为数据集"""
    def __init__(self, num_samples=1000, seq_len=10, future_steps=30, input_dim=4, emotion_num_classes=7, distraction_num_classes=5):
        """
        初始化数据集
        
        Args:
            num_samples: 样本数量
            seq_len: 历史轨迹序列长度
            future_steps: 未来轨迹预测步数
            input_dim: 轨迹特征维度
            emotion_num_classes: 情绪类别数量
            distraction_num_classes: 分心类别数量
        """
        self.emotion_dataset = EmotionDataset(num_samples, emotion_num_classes)
        self.distraction_dataset = DistractionDataset(num_samples, distraction_num_classes)
        self.trajectory_dataset = TrajectoryDataset(num_samples, seq_len, future_steps, input_dim)
    
    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.emotion_dataset)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            x_e: 情绪特征 [3, 224, 224]
            label_e: 情绪类别标签 [1]
            x_d: 分心特征 [3, 224, 224]
            label_d: 分心类别标签 [1]
            x_t: 历史轨迹 [seq_len, input_dim]
            y: 未来轨迹 [future_steps, input_dim]
        """
        # 获取情绪数据
        x_e, label_e = self.emotion_dataset[idx]
        
        # 获取分心数据
        x_d, label_d = self.distraction_dataset[idx]
        
        # 获取轨迹数据
        x_t, y = self.trajectory_dataset[idx]
        
        return x_e, label_e, x_d, label_d, x_t, y

def get_dataloader(num_samples=1000, batch_size=32, shuffle=True, seq_len=10, future_steps=30, input_dim=4, emotion_num_classes=7, distraction_num_classes=5):
    """
    获取数据加载器
    
    Args:
        num_samples: 样本数量
        batch_size: 批次大小
        shuffle: 是否打乱数据
        seq_len: 历史轨迹序列长度
        future_steps: 未来轨迹预测步数
        input_dim: 轨迹特征维度
        emotion_num_classes: 情绪类别数量
        distraction_num_classes: 分心类别数量
        
    Returns:
        dataloader: 数据加载器
    """
    dataset = DrivingBehaviorDataset(num_samples, seq_len, future_steps, input_dim, emotion_num_classes, distraction_num_classes)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    return dataloader