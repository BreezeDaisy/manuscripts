import os
import logging
from datetime import datetime


class Logger:
    """
    训练日志工具
    """
    def __init__(self, log_dir='logs'):
        """
        初始化日志工具
        
        Args:
            log_dir: 日志保存目录
        """
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'train_{current_time}.log')
        
        # 配置日志
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            
            # 添加处理器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_epoch(self, epoch, total_epochs, train_loss, val_loss=None, lr=None):
        """
        记录每个epoch的训练信息
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
            train_loss: 训练损失
            val_loss: 验证损失（可选）
            lr: 学习率（可选）
        """
        log_message = f"Epoch [{epoch}/{total_epochs}]"
        log_message += f" | Train Loss: {train_loss:.4f}"
        
        if val_loss is not None:
            log_message += f" | Val Loss: {val_loss:.4f}"
        
        if lr is not None:
            log_message += f" | LR: {lr:.6f}"
        
        self.logger.info(log_message)
    
    def log_info(self, message):
        """
        记录一般信息
        
        Args:
            message: 信息内容
        """
        self.logger.info(message)
    
    def log_error(self, message):
        """
        记录错误信息
        
        Args:
            message: 错误信息
        """
        self.logger.error(message)

    # 定义参数统计函数（核心）
def count_model_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    统计模型参数总量
    Args:
        model: PyTorch模型实例
        trainable_only: 是否只统计可训练参数（默认True，推荐）
    Returns:
        total_params: 参数总数
    """
    if trainable_only:
        # 只统计可训练参数（推荐，排除冻结层）
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        # 统计所有参数（包括冻结层、BN层的running_mean等）
        total_params = sum(p.numel() for p in model.parameters())
    return total_params

# 格式化输出函数（转成K/M/B量级，更易读）
def format_params_count(params_count: int) -> str:
    """格式化参数数量：1234 → 1.23K，1234567 → 1.23M"""
    if params_count < 1000:
        return f"{params_count} params"
    elif params_count < 1_000_000:
        return f"{params_count / 1000:.2f}K params"
    elif params_count < 1_000_000_000:
        return f"{params_count / 1_000_000:.2f}M params"
    else:
        return f"{params_count / 1_000_000_000:.2f}B params"