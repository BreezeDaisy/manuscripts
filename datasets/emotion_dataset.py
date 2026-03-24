import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# 定义数据变换
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

"""
#=========================================================================#
# 加载训练集
train_dataset = EmotionDataset(
    dataset_dir="/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/Constructed_Small_sample_0.85",  # 这里可以修改数据集地址
    split="train"
)

# 加载验证集
val_dataset = EmotionDataset(
    dataset_dir="/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/Constructed_Small_sample_0.85",  # 这里可以修改数据集地址
    split="val"
)
#=========================================================================#
"""


class EmotionDataset(Dataset):
    """情绪数据集"""
    def __init__(self, dataset_dir, split="train", transform=TRANSFORM):
        """
        初始化情绪数据集
        Args:
            dataset_dir: 数据集根目录
            split: 数据集划分，"train" 或 "val"
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.emotion_labels = {
            "antipathic": 0,
            "fear": 1,
            "happy": 2,
            "neutral": 3,
            "sad": 4,
            "surprise": 5
        }
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """
        从文件系统加载数据
        """
        split_dir = os.path.join(self.dataset_dir, self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Directory {split_dir} does not exist")
        
        for emotion, label in self.emotion_labels.items():
            emotion_dir = os.path.join(split_dir, emotion)
            if not os.path.exists(emotion_dir):
                continue
            
            for filename in os.listdir(emotion_dir):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(emotion_dir, filename)
                    self.image_paths.append(img_path)
                    self.labels.append(label)
    
    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取数据项
        Args:
            idx: 数据索引
        Returns:
            image: 预处理后的图像
            label: 情绪类别标签
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert("RGB")
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == "__main__":
    # 测试数据集加载
    train_dataset = EmotionDataset(
        dataset_dir="/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/Constructed_Small_sample_0.85",
        split="train"
    )
    
    val_dataset = EmotionDataset(
        dataset_dir="/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/Constructed_Small_sample_0.85",
        split="val"
    )
    # 打印数据集信息
    print("\n=== 验证数据集加载 ===")
    print(f"数据集路径: {train_dataset.dataset_dir}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")