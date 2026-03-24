import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional

class TrajectoryDataset(Dataset):
    """轨迹数据集"""
    def __init__(self, data_root, split="train", seq_len=10, future_steps=30, input_dim=4):
        """
        初始化轨迹数据集
        
        Args:
            data_root: 数据根目录
            split: 数据集划分，"train", "val" 或 "test"
            seq_len: 历史轨迹序列长度
            future_steps: 未来轨迹预测步数
            input_dim: 轨迹特征维度
        """
        self.data_root = Path(data_root)
        self.split = split
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.input_dim = input_dim
        
        # 加载数据文件列表
        self.file_list = self._load_file_list()
    
    def _load_file_list(self):
        """
        加载数据文件列表
        """
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            # 如果目录不存在，使用随机数据（用于测试）
            return []
        
        # 假设数据文件为 .pt 格式
        return sorted(list(split_dir.glob("*.pt")))
    
    def __len__(self):
        """
        返回数据集长度
        """
        # 如果没有实际数据文件，返回默认样本数
        if len(self.file_list) == 0:
            return 1000
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            history: 历史轨迹 [seq_len, input_dim]
            future: 未来轨迹 [future_steps, input_dim]
        """
        if len(self.file_list) > 0:
            # 加载实际数据
            try:
                data = torch.load(self.file_list[idx])
                # 假设数据格式为包含 history 和 future 字段
                if "history" in data and "future" in data:
                    return data["history"], data["future"]
            except Exception as e:
                print(f"加载数据失败: {e}")
        
        # 如果没有实际数据或加载失败，生成随机数据
        history = torch.randn(self.seq_len, self.input_dim)
        future = torch.randn(self.future_steps, self.input_dim)
        return history, future

class TrajectoryDataModule:
    """轨迹数据模块"""
    def __init__(
        self,
        data_root: str,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        seq_len: int = 10,
        future_steps: int = 30,
        input_dim: int = 4,
    ):
        """
        初始化轨迹数据模块
        
        Args:
            data_root: 数据根目录
            train_batch_size: 训练批次大小
            val_batch_size: 验证批次大小
            test_batch_size: 测试批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载线程数
            pin_memory: 是否使用锁页内存
            seq_len: 历史轨迹序列长度
            future_steps: 未来轨迹预测步数
            input_dim: 轨迹特征维度
        """
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.input_dim = input_dim
        
        # 初始化数据集
        self.setup()
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        设置数据集
        """
        self.train_dataset = TrajectoryDataset(
            data_root=self.data_root,
            split="train",
            seq_len=self.seq_len,
            future_steps=self.future_steps,
            input_dim=self.input_dim
        )
        
        self.val_dataset = TrajectoryDataset(
            data_root=self.data_root,
            split="val",
            seq_len=self.seq_len,
            future_steps=self.future_steps,
            input_dim=self.input_dim
        )
        
        self.test_dataset = TrajectoryDataset(
            data_root=self.data_root,
            split="test",
            seq_len=self.seq_len,
            future_steps=self.future_steps,
            input_dim=self.input_dim
        )
    
    def train_dataloader(self):
        """
        获取训练数据加载器
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        """
        获取验证数据加载器
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        """
        获取测试数据加载器
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
