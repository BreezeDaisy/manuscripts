import torch
from torch.utils.data import Dataset, DataLoader
from .emotion_dataset import EmotionDataset
from .distraction_dataset import DistractionDataset
from .trajectory_dataset import TrajectoryDataset

class DrivingBehaviorDataset(Dataset):
    """驾驶行为数据集"""
    def __init__(self, num_samples=1000, seq_len=10, future_steps=30, input_dim=4, emotion_num_classes=7, distraction_num_classes=5, split="train"):
        """
        初始化数据集
        
        Args:
            num_samples: 样本数量
            seq_len: 历史轨迹序列长度
            future_steps: 未来轨迹预测步数
            input_dim: 轨迹特征维度
            emotion_num_classes: 情绪类别数量
            distraction_num_classes: 分心类别数量
            split: 数据集划分，"train"或"val"
        """
        # 分别使用不同的数据集目录
        emotion_dataset_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/Constructed_Small_sample_0.85"
        distraction_dataset_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/SFDDD/images"
        trajectory_dataset_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/emp"
        
        # 输出数据路径信息
        print(f"情绪数据集路径: {emotion_dataset_dir}")
        print(f"分心数据集路径: {distraction_dataset_dir}")
        print(f"轨迹数据集路径: {trajectory_dataset_dir}")
        
        # 加载情绪数据集并统计文件数量
        try:
            self.emotion_dataset = EmotionDataset(emotion_dataset_dir, split=split)
        except Exception:
            # 如果验证集不存在，使用训练集
            self.emotion_dataset = EmotionDataset(emotion_dataset_dir, split="train")
        
        # 加载分心数据集并统计文件数量
        try:
            self.distraction_dataset = DistractionDataset(distraction_dataset_dir, split=split)
        except Exception:
            # 如果验证集不存在，使用训练集
            self.distraction_dataset = DistractionDataset(distraction_dataset_dir, split="train")
        
        # 使用正确的参数顺序初始化TrajectoryDataset
        try:
            self.trajectory_dataset = TrajectoryDataset(
                data_root=trajectory_dataset_dir, 
                split=split,
                seq_len=seq_len, 
                future_steps=future_steps, 
                input_dim=input_dim
            )
        except Exception:
            # 如果验证集不存在，使用训练集
            self.trajectory_dataset = TrajectoryDataset(
                data_root=trajectory_dataset_dir, 
                split="train",
                seq_len=seq_len, 
                future_steps=future_steps, 
                input_dim=input_dim
            )
        
        # 打印数据集信息
        print(f"{split}集 - 情绪: {len(self.emotion_dataset)}, 分心: {len(self.distraction_dataset)}, 轨迹: {len(self.trajectory_dataset)}")
    
    def __len__(self):
        """
        返回数据集长度
        根据三个数据集各自的情况，返回轨迹数据集的长度
        因为轨迹预测是主要任务，且轨迹数据集通常最大
        """
        return len(self.trajectory_dataset)
    
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
            trajectory_data: 轨迹数据字典
            y: 未来轨迹 [future_steps, input_dim]
        """
        # 使用取模运算循环访问较小的数据集
        emotion_idx = idx % len(self.emotion_dataset)
        distraction_idx = idx % len(self.distraction_dataset)
        trajectory_idx = idx % len(self.trajectory_dataset)
        
        # 获取情绪数据
        x_e, label_e = self.emotion_dataset[emotion_idx]
        
        # 获取分心数据
        x_d, label_d = self.distraction_dataset[distraction_idx]
        
        # 获取轨迹数据
        x_t, y = self.trajectory_dataset[trajectory_idx]
        
        # 构建轨迹数据字典 - 返回单个样本的形状
        num_agents = 1
        history_steps = 50
        num_lanes = 5
        lane_points = 20
        
        trajectory_data = {
            "x": torch.randn(num_agents, history_steps, 2),  # 车辆历史轨迹的位置数据 [num_agents, history_steps, 2]
            "x_padding_mask": torch.zeros(num_agents, history_steps, dtype=torch.bool),  # 轨迹填充掩码 [num_agents, history_steps]
            "x_key_padding_mask": torch.zeros(num_agents, dtype=torch.bool),  # 轨迹关键填充掩码 [num_agents]
            "x_velocity_diff": torch.randn(num_agents, history_steps),  # 速度差异 [num_agents, history_steps]
            "x_centers": torch.randn(num_agents, 2),  # 轨迹中心点 [num_agents, 2]
            "x_angles": torch.randn(num_agents, history_steps + 1),  # 轨迹角度 [num_agents, history_steps+1]
            "x_attr": torch.randint(0, 4, (num_agents, 3)),  # 轨迹属性（包括车辆类型等） [num_agents, 3]
            "lane_positions": torch.randn(num_lanes, lane_points, 2),  # 车道位置 [num_lanes, lane_points, 2]
            "lane_centers": torch.randn(num_lanes, 2),  # 车道中心点 [num_lanes, 2]
            "lane_angles": torch.randn(num_lanes),  # 车道角度 [num_lanes]
            "lane_padding_mask": torch.zeros(num_lanes, lane_points, dtype=torch.bool),  # 车道填充掩码 [num_lanes, lane_points]
            "lane_key_padding_mask": torch.zeros(num_lanes, dtype=torch.bool),  # 车道关键填充掩码 [num_lanes]
        }
        
        return x_e, label_e, x_d, label_d, trajectory_data, y

def get_dataloader(num_samples=1000, batch_size=32, shuffle=True, seq_len=10, future_steps=30, input_dim=4, emotion_num_classes=7, distraction_num_classes=5, split="train"):
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
        split: 数据集划分，"train"或"val"
        
    Returns:
        dataloader: 数据加载器
    """
    dataset = DrivingBehaviorDataset(num_samples, seq_len, future_steps, input_dim, emotion_num_classes, distraction_num_classes, split=split)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=16,  # 多线程数据加载
        pin_memory=True,  # 固定内存，加速数据传输到GPU
        prefetch_factor=4,  # 预取因子，提前加载数据
        persistent_workers=True  # 保持工作进程持久化，避免重复创建
    )
    return dataloader