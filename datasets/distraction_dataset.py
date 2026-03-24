import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class DistractionDataset(Dataset):
    """
    SFDDD数据集加载器
    SFDDD (State Farm Distracted Driver Detection) 数据集包含10个类别
    类别标签: c0 - 安全驾驶, c1 - 右手打字, c2 - 右手打电话, c3 - 左手打字, 
             c4 - 左手打电话, c5 - 调收音机, c6 - 喝水, c7 - 拿后面的东西, 
             c8 - 整理头发/化妆, c9 - 和乘客说话
    """
    def __init__(self, root_dir, split='train', transform=None, image_size=224):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # 类别映射
        self.class_names = {
            'c0': '安全驾驶',
            'c1': '右手打字',
            'c2': '右手打电话',
            'c3': '左手打字',
            'c4': '左手打电话',
            'c5': '调收音机',
            'c6': '喝水',
            'c7': '拿后面的东西',
            'c8': '整理头发/化妆',
            'c9': '和乘客说话'
        }
        
        # 类别到索引的映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names.keys())}
        
        # 获取图像路径和标签
        self.image_paths = []
        self.labels = []
        self._load_data()
        
        # 默认数据增强和预处理
        if self.transform is None:
            if self.split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((image_size,image_size)), # 输入images的尺寸要求为224x224
                    transforms.RandomHorizontalFlip(p=0.5), # 以p=0.5的概率随机水平翻转
                    transforms.RandomRotation(10), # 随机旋转角度的范围
                    transforms.ColorJitter(brightness=0.2, contrast=0.2,saturation=0.2), # 亮度(0.8,1.2)\对比度(0.8,1.2)\饱和度(0.8,1.2)随机调整
                    transforms.ToTensor(), # 将输入转换为Tensor对象,将通道从[H,W,C]转换为[C,H,W],归一化至[0.0,1.0]
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化至[-2.1,2.6]
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    
    def _load_data(self):
        """加载数据集"""
        # 确定使用的目录
        split_dir = os.path.join(self.root_dir, self.split)
        
        # 检查目录是否存在
        if not os.path.exists(split_dir):
            print(f"警告: 目录 {split_dir} 不存在")
            return
        
        # 遍历类别文件夹
        class_folders = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        if not class_folders:
            print(f"警告: 目录 {split_dir} 中没有找到类别文件夹")
            return
        
        # 加载图像和标签
        for class_name in class_folders:
            class_dir = os.path.join(split_dir, class_name)
            
            # 检查类别是否在映射中
            if class_name not in self.class_to_idx:
                print(f"警告: 类别 {class_name} 不在预定义的类别映射中")
                continue
            
            # 遍历图像文件
            img_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]
            
            for img_name in img_files:
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"成功加载 {len(self.image_paths)} 张图像到 {self.split} 数据集")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空白图像和标签
            # image = Image.new('RGB', (self.image_size, self.image_size))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloaders(config):
    """
    获取数据加载器
    """
    # 创建数据加载器
    train_dataset = DistractionDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        image_size=config['data']['image_size']
    )
    
    val_dataset = DistractionDataset(
        root_dir=config['data']['root_dir'],
        split='val',
        image_size=config['data']['image_size']
    )
    
    test_dataset = DistractionDataset(
        root_dir=config['data']['root_dir'],
        split='test',
        image_size=config['data']['image_size']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True, # 在每个epoch中打乱数据顺序
        num_workers=config['data']['num_workers'],
        pin_memory=True # 锁页CPU内存，加快与GPU的数据传输
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_names