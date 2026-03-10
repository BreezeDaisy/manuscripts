import torch
from torch.utils.data import Dataset


class DistractionDataset(Dataset):
    """分心数据集"""
    def __init__(self, num_samples=1000, num_classes=5):
        """
        初始化分心数据集
        
        Args:
            num_samples: 样本数量
            num_classes: 分心类别数量
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
    
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
            image: 随机生成的分心相关图像 [3, 224, 224]
            label: 分心类别标签 [1]
        """
        # 生成随机图像数据
        image = torch.randn(3, 224, 224)
        
        # 生成随机标签
        label = torch.randint(0, self.num_classes, (1,)).squeeze()
        
        return image, label