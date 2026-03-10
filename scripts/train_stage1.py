import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.full_model import DrivingBehaviorModel
from trainer.trainer import Trainer
from utils.seed import set_seed
from utils.logger import Logger
from configs.config import model_config, train_config, data_config, loss_config, other_config

# 合并配置
config = {
    'model_config': model_config,
    'train_config': train_config,
    'loss_config': loss_config,
    'data_config': data_config,
    'other_config': other_config
}

# 设置随机种子
set_seed(42)

# 初始化日志工具
logger = Logger(config['other_config']['log_dir'])
logger.log_info("开始 Stage1: 情绪和分心编码器训练")

# 初始化模型
model = DrivingBehaviorModel(config['model_config'])
logger.log_info(f"模型初始化完成，潜在维度: {config['model_config']['latent_dim']}")

# 加载Stage0的模型权重
try:
    model.load_state_dict(torch.load('checkpoints/stage0_epoch50.pth'))
    logger.log_info("成功加载Stage0模型权重")
except:
    logger.log_info("未找到Stage0模型权重，使用随机初始化")

# 初始化训练器
trainer = Trainer(model, config)

# 执行Stage1训练
trainer.train_stage1()

logger.log_info("Stage1: 情绪和分心编码器训练完成")