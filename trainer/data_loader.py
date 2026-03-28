import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets.trajectory_dataset import TrajectoryDataset


def get_trajectory_loader(dataset_dir, split, batch_size, num_workers=8):
    """
    获取轨迹数据集的数据加载器 - 完全按照emp.py的方式
    
    Args:
        dataset_dir: 数据集根目录
        split: 数据集划分，"train", "val" 或 "test"
        batch_size: 批次大小
        num_workers: 数据加载线程数
        
    Returns:
        DataLoader: 轨迹数据集的数据加载器
    """
    dataset = TrajectoryDataset(dataset_dir, split=split)
    
    # 根据num_workers设置DataLoader参数
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            collate_fn=collate_trajectory_data
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_trajectory_data
        )


def collate_trajectory_data(batch):
    """
    轨迹数据的collate函数 - 完全复刻av2_dataset.py的实现
    
    Args:
        batch: 数据批次
        
    Returns:
        整理后的数据字典
    """
    data = {}

    # 1. 填充序列数据（使用pad_sequence）
    for key in [
        "x",
        "x_attr",
        "x_positions",
        "x_centers",
        "x_angles",
        "x_velocity",
        "x_velocity_diff",
        "lane_positions",
        "lane_centers",
        "lane_angles",
        "lane_attr",
        "is_intersections",
    ]:
        if key in batch[0]:
            data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

    # 2. 可选键处理
    if "x_scored" in batch[0]:
        data["x_scored"] = pad_sequence(
            [b["x_scored"] for b in batch], batch_first=True
        )

    if batch[0].get("y") is not None:
        data["y"] = pad_sequence([b["y"] for b in batch], batch_first=True)

    # 3. 掩码填充（padding_value=True表示无效位置）
    for key in ["x_padding_mask", "lane_padding_mask"]:
        if key in batch[0]:
            data[key] = pad_sequence(
                [b[key] for b in batch], batch_first=True, padding_value=True
            )

    # 4. 生成关键填充掩码（用于Transformer）
    if "x_padding_mask" in data:
        data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)  # [B, N]
    if "lane_padding_mask" in data:
        data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)  # [B, M]
    
    # 5. 统计有效数量
    if "x_key_padding_mask" in data:
        data["num_actors"] = (~data["x_key_padding_mask"]).sum(-1)  # [B]
    if "lane_key_padding_mask" in data:
        data["num_lanes"] = (~data["lane_key_padding_mask"]).sum(-1)  # [B]

    # 6. 元数据（直接列表收集）
    if "scenario_id" in batch[0]:
        data["scenario_id"] = [b["scenario_id"] for b in batch]
    if "track_id" in batch[0]:
        data["track_id"] = [b["track_id"] for b in batch]

    # 7. 坐标变换参数
    if "origin" in batch[0]:
        data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)  # [B, 2]
    if "theta" in batch[0]:
        data["theta"] = torch.cat([b["theta"] for b in batch])  # [B]

    return data


def get_emotion_loader(dataset_dir, split, batch_size, num_workers=8):
    """
    获取情绪数据集的数据加载器
    
    Args:
        dataset_dir: 数据集根目录
        split: 数据集划分，"train" 或 "val"
        batch_size: 批次大小
        num_workers: 数据加载线程数
        
    Returns:
        DataLoader: 情绪数据集的数据加载器
    """
    from datasets.emotion_dataset import EmotionDataset
    dataset = EmotionDataset(dataset_dir, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )


def get_distraction_loader(dataset_dir, split, batch_size, num_workers=8):
    """
    获取分心数据集的数据加载器
    
    Args:
        dataset_dir: 数据集根目录
        split: 数据集划分，"train", "val" 或 "test"
        batch_size: 批次大小
        num_workers: 数据加载线程数
        
    Returns:
        DataLoader: 分心数据集的数据加载器
    """
    from datasets.distraction_dataset import DistractionDataset
    dataset = DistractionDataset(dataset_dir, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )


def move_trajectory_data_to_device(data, device):
    """
    将轨迹数据移动到指定设备
    
    Args:
        data: 轨迹数据字典
        device: 设备
        
    Returns:
        dict: 移动到设备后的轨迹数据字典
    """
    # 遍历数据字典，将所有张量移动到设备
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
    return data


def create_minimal_trajectory_data_dict(batch_size, device):
    """
    创建最小化的轨迹数据字典（仅用于Stage1模型调用）
    注意：此函数仅在Stage1训练情绪和分心模型时使用，
    因为Stage1不需要真实轨迹数据，但模型架构需要trajectory_data参数
    
    Args:
        batch_size: 批次大小
        device: 设备
        
    Returns:
        dict: 最小化的轨迹数据字典
    """
    return {
        "x": torch.zeros(batch_size, 1, 50, 2, device=device),
        "x_padding_mask": torch.zeros(batch_size, 1, 50, dtype=torch.bool, device=device),
        "x_key_padding_mask": torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
        "x_velocity_diff": torch.zeros(batch_size, 1, 50, device=device),
        "x_centers": torch.zeros(batch_size, 1, 2, device=device),
        "x_angles": torch.zeros(batch_size, 1, 110, device=device),
        "x_attr": torch.zeros(batch_size, 1, 3, device=device, dtype=torch.int),
        "lane_positions": torch.zeros(batch_size, 1, 20, 2, device=device),
        "lane_centers": torch.zeros(batch_size, 1, 2, device=device),
        "lane_angles": torch.zeros(batch_size, 1, device=device),
        "lane_padding_mask": torch.zeros(batch_size, 1, 20, dtype=torch.bool, device=device),
        "lane_key_padding_mask": torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
    }
