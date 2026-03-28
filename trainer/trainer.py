import os
import warnings

# 忽略特定的警告
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_utilities")

# 在导入任何模块之前设置环境变量，忽略tensorflow的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import torch
import torch.optim as optim

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from losses.total_loss import TotalLoss
from .data_loader import create_minimal_trajectory_data_dict
from .training_stages import Stage0Trainer, Stage1Trainer, Stage2Trainer, Stage3Trainer
from .model_utils import save_model, load_model


class Trainer:
    """训练器"""
    def __init__(self, model, config):
        """
        初始化训练器
        Args:
            model: 模型
            config: 配置参数
        """
        self.model = model
        self.config = config
        # 使用单GPU进行训练 - 指定使用GPU 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 指定使用GPU 0
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        print(f"使用{self.device}进行训练")
        
        self.model.to(self.device)
        
        # 损失函数权重
        self.loss_fn = TotalLoss(
            lambda_traj=config['loss_config']['lambda_traj'],
            lambda_kl_e=config['loss_config']['lambda_kl_e'],
            lambda_kl_d=config['loss_config']['lambda_kl_d'],
            lambda_orth=config['loss_config']['lambda_orth']
        )
        
        # 优化器 - 使用AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['train_config']['learning_rate'],
            weight_decay=config['train_config']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['train_config']['epochs']['stage0'],
            eta_min=config['train_config']['learning_rate'] * 0.01  # 最小学习率
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        # 初始化损失记录
        self.loss_history = {
            'stage0': {'train_loss': [], 'val_loss': []},
            'stage1': {'train_loss': [], 'val_loss': []},
            'stage2': {'train_loss': [], 'val_loss': []}
        }
        
        # 初始化最佳验证损失
        self.best_val_loss = {
            'stage0': float('inf'),
            'stage1': float('inf'),
            'stage2': float('inf')
        }
    
    def train_stage0(self):
        """
        Stage0: 轨迹网络预训练
        只使用emp数据集
        """
        stage0_trainer = Stage0Trainer(
            self.model, self.device, self.loss_fn, 
            self.optimizer, self.scheduler, self.config
        )
        stage0_trainer.train(
            self.loss_history, 
            self.best_val_loss, 
            lambda filename: save_model(self.model, self.config, filename),
            lambda stage: self.visualize_loss(stage)
        )
    
    def train_stage1(self):
        """
        Stage1: 情绪和分心编码器训练
        只使用情绪和分心数据集，不使用emp数据集
        """
        stage1_trainer = Stage1Trainer(
            self.model, self.device, self.loss_fn, self.config
        )
        stage1_trainer.train(
            self.loss_history, 
            self.best_val_loss, 
            lambda filename: save_model(self.model, self.config, filename),
            lambda filename: load_model(self.model, self.config, filename),
            lambda stage: self.visualize_loss(stage)
        )
    
    def train_stage2(self):
        """
        Stage2: FiLM调制训练
        只使用emp数据集，冻结情绪和分心模型
        """
        stage2_trainer = Stage2Trainer(
            self.model, self.device, self.loss_fn, 
            self.optimizer, self.config
        )
        stage2_trainer.train(
            self.loss_history, 
            self.best_val_loss, 
            lambda filename: save_model(self.model, self.config, filename),
            lambda stage: self.visualize_loss(stage)
        )
    
    def train_stage3(self):
        """
        Stage3: 联合微调训练
        加载轨迹模型、情绪模型、分心模型、FiLM权重，以更小的学习率完成整个流程的权重微调
        """
        stage3_trainer = Stage3Trainer(
            self.model, self.device, self.loss_fn, self.config
        )
        stage3_trainer.train(
            self.loss_history, 
            self.best_val_loss, 
            lambda filename: save_model(self.model, self.config, filename),
            lambda filename: load_model(self.model, self.config, filename),
            lambda stage: self.visualize_loss(stage)
        )
    
    def evaluate(self):
        """
        评估模型
        
        Returns:
            val_loss: 验证损失
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                x_e, label_e, x_d, label_d, trajectory_data, y = batch
                
                # 将数据移动到设备
                x_e = x_e.to(self.device)
                label_e = label_e.to(self.device)
                x_d = x_d.to(self.device)
                label_d = label_d.to(self.device)
                y = y.to(self.device)
                
                # 将轨迹数据移动到设备
                for key in trajectory_data:
                    if isinstance(trajectory_data[key], torch.Tensor):
                        trajectory_data[key] = trajectory_data[key].to(self.device)
                
                # 检查并裁剪标签范围
                # 获取模型（考虑DataParallel的情况）
                model = self.model.module if hasattr(self.model, 'module') else self.model
                
                # 检查情绪标签范围
                if label_e.max() >= model.emotion_encoder.num_classes:
                    print(f"警告: 情绪标签超出范围: {label_e.max()} >= {model.emotion_encoder.num_classes}")
                    # 裁剪标签到有效范围
                    label_e = torch.clamp(label_e, 0, model.emotion_encoder.num_classes - 1)
                
                # 检查分心标签范围
                if label_d.max() >= model.distraction_encoder.num_classes:
                    print(f"警告: 分心标签超出范围: {label_d.max()} >= {model.distraction_encoder.num_classes}")
                    # 裁剪标签到有效范围
                    label_d = torch.clamp(label_d, 0, model.distraction_encoder.num_classes - 1)
                
                # 前向传播
                pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(x_e, label_e, x_d, label_d, trajectory_data)
                
                # 计算总损失
                total_loss_batch, _, _, _, _, _, _, _, _, _ = self.loss_fn(pred_dict['y_hat'], y, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_dict['pi'])
                total_loss += total_loss_batch.item()
        
        return total_loss / len(self.val_loader)
    
    def visualize_loss(self, stage):
        """
        调用可视化模块进行损失曲线和评价指标的可视化
        
        Args:
            stage: 训练阶段名称
        """
        from .visualize import visualize_loss
        visualize_loss(stage, self.loss_history, self.config['other_config']['log_dir'])


# 导入配置
from configs.config import model_config, train_config, data_config, loss_config, other_config
config = {
    'model_config': model_config,
    'train_config': train_config,
    'loss_config': loss_config,
    'data_config': data_config,
    'other_config': other_config
}

# 导入本地模型
from models.full_model import DrivingBehaviorModel


def main():
    """主训练函数"""
    # 检查是否使用GPU
    use_cuda = torch.cuda.is_available()
    
    # 在GPU上禁用确定性算法以避免CUDA错误
    if not use_cuda:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='Driver State Trajectory Training')
    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2, 3, 4], help='训练阶段 (4表示按顺序训练所有阶段)')
    parser.add_argument('--config', type=str, default='configs/config.py', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='从指定模型恢复训练')
    args = parser.parse_args()
    
    # 导入配置
    import importlib.util
    spec = importlib.util.spec_from_file_location('config', args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # 构建配置字典
    config = {
        'model_config': config_module.model_config,
        'train_config': config_module.train_config,
        'loss_config': config_module.loss_config,
        'data_config': config_module.data_config,
        'other_config': config_module.other_config
    }
    
    # 创建模型实例
    model = DrivingBehaviorModel(config['model_config'])
    
    # 初始化训练器
    trainer = Trainer(model, config)
    
    # 如果指定了恢复点，加载模型
    if args.resume:
        load_model(model, config, args.resume)
    
    # 根据指定的阶段进行训练
    try:
        if args.stage == 0:
            print("开始训练 Stage 0: 轨迹预训练")
            trainer.train_stage0()
        elif args.stage == 1:
            print("开始训练 Stage 1: 情绪和分心编码器训练")
            trainer.train_stage1()
        elif args.stage == 2:
            print("开始训练 Stage 2: FiLM训练")
            trainer.train_stage2()
        elif args.stage == 3:
            print("开始训练 Stage 3: 联合微调训练")
            trainer.train_stage3()
        elif args.stage == 4:
            print("开始按顺序训练所有阶段")
            # Stage 0: 轨迹预训练
            print("\n===== 开始训练 Stage 0: 轨迹预训练 =====")
            trainer.train_stage0()
            
            # Stage 1: 情绪和分心编码器训练
            print("\n===== 开始训练 Stage 1: 情绪和分心编码器训练 =====")
            trainer.train_stage1()
            
            # Stage 2: FiLM训练
            print("\n===== 开始训练 Stage 2: FiLM训练 =====")
            trainer.train_stage2()
            
            # Stage 3: 联合微调训练
            print("\n===== 开始训练 Stage 3: 联合微调训练 =====")
            trainer.train_stage3()
            
            print("\n所有阶段训练完成！")
    except KeyboardInterrupt:
        print("训练被用户中断")
    finally:
        # 清理资源
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()