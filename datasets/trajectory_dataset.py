import torch
from torch.utils.data import Dataset
from pathlib import Path


class TrajectoryDataset(Dataset):
    """
    轨迹数据集 - 按照emp.py的方式加载数据
    每个数据文件包含一个场景的所有智能体和车道信息
    """
    def __init__(self, data_root, split="train"):
        """
        初始化轨迹数据集
        
        Args:
            data_root: 数据根目录
            split: 数据集划分，"train", "val" 或 "test"
        """
        super(TrajectoryDataset, self).__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.data_folder = self.data_root / split
        
        # 加载数据文件列表
        self.file_list = sorted(list(self.data_folder.glob("*.pt")))
        
        if len(self.file_list) == 0:
            raise ValueError(f"没有找到数据文件: {self.data_folder}")
        
        print(f"数据根目录: {data_root}/{split}, 文件总数: {len(self.file_list)}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        """
        获取数据项 - 直接加载.pt文件，不做任何修改
        数据应该已经是预处理好的格式，与emp.py兼容
        """
        data = torch.load(self.file_list[index])
        return data
