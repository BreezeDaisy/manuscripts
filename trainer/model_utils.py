import os
import torch


def save_model(model, config, filename):
    """
    保存模型
    
    Args:
        model: 模型实例
        config: 配置字典
        filename: 文件名
    """
    # 获取原始模型（考虑DataParallel的情况）
    model = model.module if hasattr(model, 'module') else model
    save_path = os.path.join(config['other_config']['checkpoint_dir'], filename)
    torch.save(model.state_dict(), save_path)
    print(f"模型保存到 {save_path}")


def load_model(model, config, filename):
    """
    加载模型
    
    Args:
        model: 模型实例
        config: 配置字典
        filename: 文件名
    """
    # 获取原始模型（考虑DataParallel的情况）
    model = model.module if hasattr(model, 'module') else model
    load_path = os.path.join(config['other_config']['checkpoint_dir'], filename)
    model.load_state_dict(torch.load(load_path))
    print(f"从 {load_path} 加载模型")
