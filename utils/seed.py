import torch
import numpy as np
import random


def set_seed(seed):
    """
    设置随机种子，确保实验可重现
    
    Args:
        seed: 随机种子值
    """
    # 设置Python随机种子
    random.seed(seed)
    
    # 设置NumPy随机种子
    np.random.seed(seed)
    
    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    
    # 如果使用GPU，设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 确保CUDA卷积操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False