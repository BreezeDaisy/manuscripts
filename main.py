import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
logger.log_info("开始完整训练流程")

# 初始化模型
model = DrivingBehaviorModel(config['model_config'])
# 模型参数量统计
# 统计参数并输出到日志
    total_params = count_model_parameters(model)
    formatted_params = format_params_count(total_params)
    
    logger.info(f"模型可训练参数总量: {formatted_params} (原始数值: {total_params})")
    # 若需要统计所有参数（包括冻结层）
    total_all_params = count_model_parameters(model, trainable_only=False)
    logger.info(f"模型所有参数总量: {format_params_count(total_all_params)} (原始数值: {total_all_params})")

logger.log_info(f"模型初始化完成，潜在维度: {config['model_config']['latent_dim']}")

# 初始化训练器
trainer = Trainer(model, config)

# 执行四个训练阶段
logger.log_info("\n=== Stage0: 轨迹网络预训练 ===")
trainer.train_stage0()

logger.log_info("\n=== Stage1: 情绪和分心编码器训练 ===")
trainer.train_stage1()

logger.log_info("\n=== Stage2: FiLM调制训练 ===")
trainer.train_stage2()

logger.log_info("\n=== Stage3: 联合微调 ===")
trainer.train_stage3()

logger.log_info("\n完整训练流程完成！")