from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader

from .av2_dataset import Av2Dataset, collate_fn


class Av2DataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        data_folder: str,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        test: bool = False,
    ):
        super(Av2DataModule, self).__init__()
        self.data_root = Path(data_root)
        self.data_folder = data_folder
        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = Av2Dataset(
                data_root=self.data_root / self.data_folder, cached_split="train"
            )
            self.val_dataset = Av2Dataset(
                data_root=self.data_root / self.data_folder, cached_split="val"
            )
        else:
            self.test_dataset = Av2Dataset(
                data_root=self.data_root / self.data_folder, cached_split="test"
            )

    def train_dataloader(self):
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

if __name__ == "__main__":
    # 使用正确的数据路径
    data_root = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/emp"
    data_folder = ""  # 数据直接在 train/val/test 目录下
    datamodule = Av2DataModule(
        data_root=data_root,
        data_folder=data_folder,
        train_batch_size=32,
        val_batch_size=32,
        test_batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        test=True,  # 启用测试集加载
    )
    datamodule.setup()
    
    def print_sample_info(dataset_name, dataset):
        """打印数据集样本信息"""
        print(f"\n{dataset_name}:")
        print(f"  样本数: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[0]
            if isinstance(sample, dict):
                print(f"  样本键: {list(sample.keys())}")
                for key, value in sample.items():
                    if hasattr(value, 'shape'):
                        print(f"    {key}: shape={value.shape}")
                    elif isinstance(value, (list, tuple)):
                        print(f"    {key}: len={len(value)}, type={type(value[0]).__name__ if value else 'empty'}")
                    else:
                        print(f"    {key}: type={type(value).__name__}, value={value}")
            elif hasattr(sample, 'shape'):
                print(f"  样本shape: {sample.shape}")
            else:
                print(f"  样本内容: {sample}")
    
    if hasattr(datamodule, 'train_dataset') and datamodule.train_dataset is not None:
        print_sample_info("训练集", datamodule.train_dataset)
    if hasattr(datamodule, 'val_dataset') and datamodule.val_dataset is not None:
        print_sample_info("验证集", datamodule.val_dataset)
    if hasattr(datamodule, 'test_dataset') and datamodule.test_dataset is not None:
        print_sample_info("测试集", datamodule.test_dataset)
