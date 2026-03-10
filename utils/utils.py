import torch
import numpy as np
import os
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"设置随机种子为 {seed}")

def save_checkpoint(model, optimizer, epoch, save_path):
    """
    保存检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 训练轮数
        save_path: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)
    logger.info(f"检查点保存到 {save_path}")

def load_checkpoint(model, optimizer, load_path):
    """
    加载检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        load_path: 加载路径
        
    Returns:
        epoch: 训练轮数
    """
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    logger.info(f"从 {load_path} 加载检查点，当前轮数: {epoch}")
    return epoch

def compute_metrics(pred, target):
    """
    计算评估指标
    
    Args:
        pred: 预测的轨迹 [batch_size, future_steps, output_dim]
        target: 真实的轨迹 [batch_size, future_steps, output_dim]
        
    Returns:
        metrics: 评估指标字典
    """
    # 计算MSE
    mse = torch.mean((pred - target) ** 2).item()
    
    # 计算RMSE
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    
    # 计算MAE
    mae = torch.mean(torch.abs(pred - target)).item()
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }
    
    return metrics

def create_directory(path):
    """
    创建目录
    
    Args:
        path: 目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"创建目录: {path}")
    else:
        logger.info(f"目录已存在: {path}")