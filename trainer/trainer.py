import os
import time
import warnings
import matplotlib.pyplot as plt

# 忽略特定的警告
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_utilities")

# 在导入任何模块之前设置环境变量，忽略tensorflow的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
import pytorch_lightning as pl
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (
    LearningRateMonitor, ModelCheckpoint, RichModelSummary, RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# 忽略tensorflow的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from losses.total_loss import TotalLoss
from datasets.dataset import get_dataloader

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
    
    def train_stage0(self):
        """
        Stage0: 轨迹网络预训练
        只使用emp数据集
        """
        print("开始 Stage0: 轨迹网络预训练")
        
        # 获取模型（考虑DataParallel的情况）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 加载情绪模型预训练权重
        emotion_pretrained_path = "/home/zdx/python_daima/MVim/manuscripts/checkpoints/face/best_model.pth"
        if os.path.exists(emotion_pretrained_path):
            try:
                emotion_state_dict = torch.load(emotion_pretrained_path)
                # 加载权重到情绪编码器
                model.emotion_encoder.load_state_dict(emotion_state_dict, strict=False)
                print(f"成功加载情绪模型预训练权重: {emotion_pretrained_path}")
            except Exception as e:
                print(f"加载情绪模型权重失败: {e}")
        else:
            print(f"情绪模型预训练权重文件不存在: {emotion_pretrained_path}")
        
        # 加载分心模型预训练权重
        distraction_pretrained_path = "/home/zdx/python_daima/MVim/manuscripts/checkpoints/pose/best_model.pth"
        if os.path.exists(distraction_pretrained_path):
            try:
                distraction_state_dict = torch.load(distraction_pretrained_path)
                # 加载权重到分心编码器
                model.distraction_encoder.load_state_dict(distraction_state_dict, strict=False)
                print(f"成功加载分心模型预训练权重: {distraction_pretrained_path}")
            except Exception as e:
                print(f"加载分心模型权重失败: {e}")
        else:
            print(f"分心模型预训练权重文件不存在: {distraction_pretrained_path}")
        
        # 冻结情绪和分心编码器
        for param in model.emotion_encoder.parameters():
            param.requires_grad = False
        
        for param in model.distraction_encoder.parameters():
            param.requires_grad = False
        
        # 冻结FiLM模块
        for param in model.film_module.parameters():
            param.requires_grad = False
        
        # 加载emp数据集（只使用轨迹数据）
        from datasets.trajectory_dataset import TrajectoryDataset
        emp_dataset_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/emp"
        
        # 训练集
        train_emp_dataset = TrajectoryDataset(emp_dataset_dir, split="train", seq_len=50, future_steps=60)
        # 验证集
        val_emp_dataset = TrajectoryDataset(emp_dataset_dir, split="val", seq_len=50, future_steps=60)
        # 测试集
        test_emp_dataset = TrajectoryDataset(emp_dataset_dir, split="test", seq_len=50, future_steps=60)
        
        print(f"Stage0 - emp训练集: {len(train_emp_dataset)}, 验证集: {len(val_emp_dataset)}, 测试集: {len(test_emp_dataset)}")
        
        # 创建数据加载器
        from torch.utils.data import DataLoader
        
        emp_train_loader = DataLoader(
            train_emp_dataset,
            batch_size=self.config['train_config']['batch_size'],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        emp_val_loader = DataLoader(
            val_emp_dataset,
            batch_size=self.config['train_config']['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        emp_test_loader = DataLoader(
            test_emp_dataset,
            batch_size=self.config['train_config']['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=8,
            persistent_workers=True
        )
        
        # 训练轨迹编码器和解码器
        for epoch in range(self.config['train_config']['epochs']['stage0']):
            self.model.train()
            total_loss = 0
            
            for i, (history, future) in enumerate(tqdm(emp_train_loader, desc=f"Stage0 Epoch {epoch+1}/{self.config['train_config']['epochs']['stage0']}")):
                # 记录批次开始时间
                start_time = time.time()
                
                # 获取批次大小
                batch_size = future.shape[0]
                
                # 构建轨迹数据字典
                num_agents = 1
                trajectory_data = {
                    "x": history.unsqueeze(1).to(self.device),  # [batch_size, num_agents, history_steps, 2]
                    "x_padding_mask": torch.zeros(batch_size, num_agents, history.shape[1], dtype=torch.bool, device=self.device),
                    "x_key_padding_mask": torch.zeros(batch_size, num_agents, dtype=torch.bool, device=self.device),
                    "x_velocity_diff": torch.zeros(batch_size, num_agents, history.shape[1], device=self.device),
                    "x_centers": torch.zeros(batch_size, num_agents, 2, device=self.device),
                    "x_angles": torch.zeros(batch_size, num_agents, history.shape[1] + 1, device=self.device),
                    "x_attr": torch.zeros(batch_size, num_agents, 3, device=self.device, dtype=torch.int),
                    "lane_positions": torch.zeros(batch_size, 5, 20, 2, device=self.device),
                    "lane_centers": torch.zeros(batch_size, 5, 2, device=self.device),
                    "lane_angles": torch.zeros(batch_size, 5, device=self.device),
                    "lane_padding_mask": torch.zeros(batch_size, 5, 20, dtype=torch.bool, device=self.device),
                    "lane_key_padding_mask": torch.zeros(batch_size, 5, dtype=torch.bool, device=self.device),
                }
                
                # 将数据移动到设备
                y = future.to(self.device)
                
                try:
                    # 混合精度前向传播
                    with torch.cuda.amp.autocast():
                        # 前向传播 - 使用轨迹特征作为情绪和分心编码器的输入
                        pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(
                            trajectory_data=trajectory_data, use_trajectory_feature=True
                        )
                        
                        # 计算损失（只计算轨迹损失）
                        loss = self.loss_fn.trajectory_loss(pred_dict['y_hat'], y, pred_dict['pi'])
                    
                    # 混合精度反向传播
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    total_loss += loss.item()
                    
                    # 每100个batch打印一次损失
                    if i % 100 == 0:
                        print(f"Batch {i}/{len(emp_train_loader)} - Loss: {loss.item():.4f}")
                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # 验证
            if (epoch + 1) % self.config['other_config']['eval_interval'] == 0:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for history, future in tqdm(emp_val_loader, desc="Evaluating Stage0", leave=False):
                        # 获取批次大小
                        batch_size = future.shape[0]
                        
                        # 构建轨迹数据字典
                        num_agents = 1
                        trajectory_data = {
                            "x": history.unsqueeze(1).to(self.device),  # [batch_size, num_agents, history_steps, 2]
                            "x_padding_mask": torch.zeros(batch_size, num_agents, history.shape[1], dtype=torch.bool, device=self.device),
                            "x_key_padding_mask": torch.zeros(batch_size, num_agents, dtype=torch.bool, device=self.device),
                            "x_velocity_diff": torch.zeros(batch_size, num_agents, history.shape[1], device=self.device),
                            "x_centers": torch.zeros(batch_size, num_agents, 2, device=self.device),
                            "x_angles": torch.zeros(batch_size, num_agents, history.shape[1] + 1, device=self.device),
                            "x_attr": torch.zeros(batch_size, num_agents, 3, device=self.device, dtype=torch.int),
                            "lane_positions": torch.zeros(batch_size, 5, 20, 2, device=self.device),
                            "lane_centers": torch.zeros(batch_size, 5, 2, device=self.device),
                            "lane_angles": torch.zeros(batch_size, 5, device=self.device),
                            "lane_padding_mask": torch.zeros(batch_size, 5, 20, dtype=torch.bool, device=self.device),
                            "lane_key_padding_mask": torch.zeros(batch_size, 5, dtype=torch.bool, device=self.device),
                        }
                        
                        # 将数据移动到设备
                        y = future.to(self.device)
                        
                        # 前向传播 - 使用轨迹特征作为情绪和分心编码器的输入
                        pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(
                            trajectory_data=trajectory_data, use_trajectory_feature=True
                        )
                        
                        # 计算损失（只计算轨迹损失）
                        loss = self.loss_fn.trajectory_loss(pred_dict['y_hat'], y, pred_dict['pi'])
                        val_loss += loss.item()
                
                val_loss_avg = val_loss / len(emp_val_loader)
                train_loss_avg = total_loss / len(emp_train_loader)
                print(f"Epoch {epoch+1}, Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")
                # 记录损失
                self.loss_history['stage0']['train_loss'].append(train_loss_avg)
                self.loss_history['stage0']['val_loss'].append(val_loss_avg)
            
            # 保存模型
            if (epoch + 1) % self.config['other_config']['save_interval'] == 0:
                self.save_model(f"stage0_epoch{epoch+1}.pth")
        
        # 训练完成后可视化损失
        self.visualize_loss('stage0')
    
    def train_stage1(self):
        """
        Stage1: 情绪和分心编码器训练
        只使用情绪和分心数据集，不使用emp数据集
        """
        print("开始 Stage1: 情绪和分心编码器训练")
        
        # 获取模型（考虑DataParallel的情况）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 解冻情绪和分心编码器
        for param in model.emotion_encoder.parameters():
            param.requires_grad = True
        
        for param in model.distraction_encoder.parameters():
            param.requires_grad = True
        
        # 冻结轨迹编码器和解码器
        for param in model.trajectory_encoder.parameters():
            param.requires_grad = False
        
        for param in model.trajectory_decoder.parameters():
            param.requires_grad = False
        
        # 冻结FiLM模块
        for param in model.film_module.parameters():
            param.requires_grad = False
        
        # 加载情绪和分心数据集
        from datasets.emotion_dataset import EmotionDataset
        from datasets.distraction_dataset import DistractionDataset
        
        emotion_dataset_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/Constructed_Small_sample_0.85"
        distraction_dataset_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/SFDDD/images"
        
        # 训练集
        train_emotion_dataset = EmotionDataset(emotion_dataset_dir, split="train")
        train_distraction_dataset = DistractionDataset(distraction_dataset_dir, split="train")
        
        # 验证集
        val_emotion_dataset = EmotionDataset(emotion_dataset_dir, split="val")
        val_distraction_dataset = DistractionDataset(distraction_dataset_dir, split="val")
        
        print(f"Stage1 - 情绪训练集: {len(train_emotion_dataset)}, 验证集: {len(val_emotion_dataset)}")
        print(f"Stage1 - 分心训练集: {len(train_distraction_dataset)}, 验证集: {len(val_distraction_dataset)}")
        
        # 创建数据加载器
        from torch.utils.data import DataLoader, ConcatDataset
        
        # 为情绪和分心数据集创建单独的加载器
        emotion_train_loader = DataLoader(
            train_emotion_dataset,
            batch_size=self.config['train_config']['batch_size'],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        distraction_train_loader = DataLoader(
            train_distraction_dataset,
            batch_size=self.config['train_config']['batch_size'],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        # 为分心模型创建单独的优化器
        distraction_optimizer = optim.AdamW(
            model.distraction_encoder.parameters(),
            lr=self.config['train_config']['learning_rate'] * 1.5,  # 分心模型学习率稍高
            weight_decay=self.config['train_config']['weight_decay']
        )
        
        # 为情绪模型创建优化器
        emotion_optimizer = optim.AdamW(
            model.emotion_encoder.parameters(),
            lr=self.config['train_config']['learning_rate'],
            weight_decay=self.config['train_config']['weight_decay']
        )
        
        # 设置学习率调度器
        total_epochs = self.config['train_config']['epochs']['stage1']
        warmup_epochs = 5
        
        # 余弦退火调度器，带预热
        distraction_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            distraction_optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=self.config['train_config']['learning_rate'] * 0.01  # 最小学习率
        )
        
        emotion_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            emotion_optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=self.config['train_config']['learning_rate'] * 0.01  # 最小学习率
        )
        
        # 预热调度器
        if warmup_epochs > 0:
            warmup_distraction_scheduler = optim.lr_scheduler.LinearLR(
                distraction_optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            distraction_scheduler = optim.lr_scheduler.SequentialLR(
                distraction_optimizer,
                schedulers=[warmup_distraction_scheduler, distraction_scheduler],
                milestones=[warmup_epochs]
            )
            
            warmup_emotion_scheduler = optim.lr_scheduler.LinearLR(
                emotion_optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            emotion_scheduler = optim.lr_scheduler.SequentialLR(
                emotion_optimizer,
                schedulers=[warmup_emotion_scheduler, emotion_scheduler],
                milestones=[warmup_epochs]
            )
        
        # 训练情绪和分心编码器
        for epoch in range(self.config['train_config']['epochs']['stage1']):
            self.model.train()
            total_loss = 0
            total_kl_loss_e = 0
            total_kl_loss_d = 0
            
            # 训练情绪编码器
            print(f"训练情绪编码器...")
            for i, (x_e, label_e) in enumerate(tqdm(emotion_train_loader, desc=f"Stage1 Epoch {epoch+1}/{self.config['train_config']['epochs']['stage1']} - Emotion")):
                # 构建最小化的轨迹数据字典（仅用于模型调用）
                batch_size = x_e.shape[0]
                trajectory_data = {
                    "x": torch.randn(batch_size, 1, 50, 2, device=self.device),
                    "x_padding_mask": torch.zeros(batch_size, 1, 50, dtype=torch.bool, device=self.device),
                    "x_key_padding_mask": torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),
                    "x_velocity_diff": torch.zeros(batch_size, 1, 50, device=self.device),
                    "x_centers": torch.zeros(batch_size, 1, 2, device=self.device),
                    "x_angles": torch.zeros(batch_size, 1, 51, device=self.device),
                    "x_attr": torch.zeros(batch_size, 1, 3, device=self.device, dtype=torch.int),
                    "lane_positions": torch.zeros(batch_size, 5, 20, 2, device=self.device),
                    "lane_centers": torch.zeros(batch_size, 5, 2, device=self.device),
                    "lane_angles": torch.zeros(batch_size, 5, device=self.device),
                    "lane_padding_mask": torch.zeros(batch_size, 5, 20, dtype=torch.bool, device=self.device),
                    "lane_key_padding_mask": torch.zeros(batch_size, 5, dtype=torch.bool, device=self.device),
                }
                
                # 将数据移动到设备
                x_e = x_e.to(self.device)
                label_e = label_e.to(self.device)
                
                # 检查并裁剪标签范围
                if label_e.max() >= model.emotion_encoder.num_classes:
                    print(f"警告: 情绪标签超出范围: {label_e.max()} >= {model.emotion_encoder.num_classes}")
                    # 裁剪标签到有效范围
                    label_e = torch.clamp(label_e, 0, model.emotion_encoder.num_classes - 1)
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    # 前向传播 - 只训练情绪模型，返回分类预测
                    pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_e, pred_d = self.model(
                        x_e=x_e, label_e=label_e, x_d=None, label_d=None, 
                        trajectory_data=trajectory_data, use_trajectory_feature=False, 
                        return_classification=True
                    )
                    
                    # 计算损失：情绪分类损失（Focal Loss） + 情绪KL散度损失
                    _, traj_loss, kl_loss_e, kl_loss_d, orth_loss, cls_loss_e, cls_loss_d = self.loss_fn(
                        pred_dict['y_hat'], None, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, 
                        pred_dict['pi'], pred_e=pred_e, label_e=label_e
                    )
                    loss = cls_loss_e + kl_loss_e
                
                # 混合精度反向传播
                emotion_optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(emotion_optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                total_kl_loss_e += kl_loss_e.item()
            
            # 训练分心编码器
            print(f"训练分心编码器...")
            for i, (x_d, label_d) in enumerate(tqdm(distraction_train_loader, desc=f"Stage1 Epoch {epoch+1}/{self.config['train_config']['epochs']['stage1']} - Distraction")):
                # 构建最小化的轨迹数据字典（仅用于模型调用）
                batch_size = x_d.shape[0]
                trajectory_data = {
                    "x": torch.randn(batch_size, 1, 50, 2, device=self.device),
                    "x_padding_mask": torch.zeros(batch_size, 1, 50, dtype=torch.bool, device=self.device),
                    "x_key_padding_mask": torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),
                    "x_velocity_diff": torch.zeros(batch_size, 1, 50, device=self.device),
                    "x_centers": torch.zeros(batch_size, 1, 2, device=self.device),
                    "x_angles": torch.zeros(batch_size, 1, 51, device=self.device),
                    "x_attr": torch.zeros(batch_size, 1, 3, device=self.device, dtype=torch.int),
                    "lane_positions": torch.zeros(batch_size, 5, 20, 2, device=self.device),
                    "lane_centers": torch.zeros(batch_size, 5, 2, device=self.device),
                    "lane_angles": torch.zeros(batch_size, 5, device=self.device),
                    "lane_padding_mask": torch.zeros(batch_size, 5, 20, dtype=torch.bool, device=self.device),
                    "lane_key_padding_mask": torch.zeros(batch_size, 5, dtype=torch.bool, device=self.device),
                }
                
                # 将数据移动到设备
                x_d = x_d.to(self.device)
                label_d = label_d.to(self.device)
                
                # 检查并裁剪标签范围
                if label_d.max() >= model.distraction_encoder.num_classes:
                    print(f"警告: 分心标签超出范围: {label_d.max()} >= {model.distraction_encoder.num_classes}")
                    # 裁剪标签到有效范围
                    label_d = torch.clamp(label_d, 0, model.distraction_encoder.num_classes - 1)
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    # 前向传播 - 只训练分心模型，返回分类预测
                    pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_e, pred_d = self.model(
                        x_e=None, label_e=None, x_d=x_d, label_d=label_d, 
                        trajectory_data=trajectory_data, use_trajectory_feature=False, 
                        return_classification=True
                    )
                    
                    # 计算损失：分心分类损失（交叉熵损失） + 分心KL散度损失
                    _, traj_loss, kl_loss_e, kl_loss_d, orth_loss, cls_loss_e, cls_loss_d = self.loss_fn(
                        pred_dict['y_hat'], None, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, 
                        pred_dict['pi'], pred_d=pred_d, label_d=label_d
                    )
                    loss = cls_loss_d + kl_loss_d
                
                # 混合精度反向传播
                distraction_optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(distraction_optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                total_kl_loss_d += kl_loss_d.item()
            
            # 验证
            if (epoch + 1) % self.config['other_config']['eval_interval'] == 0:
                val_loss = 0
                val_kl_loss_e = 0
                val_kl_loss_d = 0
                
                self.model.eval()
                with torch.no_grad():
                    # 验证情绪编码器
                    for x_e, label_e in val_emotion_dataset:
                        # 构建最小化的轨迹数据字典（仅用于模型调用）
                        batch_size = 1
                        trajectory_data = {
                            "x": torch.randn(batch_size, 1, 50, 2, device=self.device),
                            "x_padding_mask": torch.zeros(batch_size, 1, 50, dtype=torch.bool, device=self.device),
                            "x_key_padding_mask": torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),
                            "x_velocity_diff": torch.zeros(batch_size, 1, 50, device=self.device),
                            "x_centers": torch.zeros(batch_size, 1, 2, device=self.device),
                            "x_angles": torch.zeros(batch_size, 1, 51, device=self.device),
                            "x_attr": torch.zeros(batch_size, 1, 3, device=self.device, dtype=torch.int),
                            "lane_positions": torch.zeros(batch_size, 5, 20, 2, device=self.device),
                            "lane_centers": torch.zeros(batch_size, 5, 2, device=self.device),
                            "lane_angles": torch.zeros(batch_size, 5, device=self.device),
                            "lane_padding_mask": torch.zeros(batch_size, 5, 20, dtype=torch.bool, device=self.device),
                            "lane_key_padding_mask": torch.zeros(batch_size, 5, dtype=torch.bool, device=self.device),
                        }
                        
                        # 将数据移动到设备
                        x_e = x_e.unsqueeze(0).to(self.device)
                        label_e = torch.tensor([label_e], device=self.device)
                        
                        with torch.cuda.amp.autocast():
                            # 前向传播 - 返回分类预测
                            pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_e, pred_d = self.model(
                                x_e=x_e, label_e=label_e, x_d=None, label_d=None, 
                                trajectory_data=trajectory_data, use_trajectory_feature=False, 
                                return_classification=True
                            )
                            # 计算损失：情绪分类损失（Focal Loss） + 情绪KL散度损失
                            _, traj_loss, kl_loss_e, kl_loss_d, orth_loss, cls_loss_e, cls_loss_d = self.loss_fn(
                                pred_dict['y_hat'], None, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, 
                                pred_dict['pi'], pred_e=pred_e, label_e=label_e
                            )
                            val_loss += (cls_loss_e + kl_loss_e).item()
                            val_kl_loss_e += kl_loss_e.item()
                    
                    # 验证分心编码器
                    for x_d, label_d in val_distraction_dataset:
                        # 构建最小化的轨迹数据字典（仅用于模型调用）
                        batch_size = 1
                        trajectory_data = {
                            "x": torch.randn(batch_size, 1, 50, 2, device=self.device),
                            "x_padding_mask": torch.zeros(batch_size, 1, 50, dtype=torch.bool, device=self.device),
                            "x_key_padding_mask": torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),
                            "x_velocity_diff": torch.zeros(batch_size, 1, 50, device=self.device),
                            "x_centers": torch.zeros(batch_size, 1, 2, device=self.device),
                            "x_angles": torch.zeros(batch_size, 1, 51, device=self.device),
                            "x_attr": torch.zeros(batch_size, 1, 3, device=self.device, dtype=torch.int),
                            "lane_positions": torch.zeros(batch_size, 5, 20, 2, device=self.device),
                            "lane_centers": torch.zeros(batch_size, 5, 2, device=self.device),
                            "lane_angles": torch.zeros(batch_size, 5, device=self.device),
                            "lane_padding_mask": torch.zeros(batch_size, 5, 20, dtype=torch.bool, device=self.device),
                            "lane_key_padding_mask": torch.zeros(batch_size, 5, dtype=torch.bool, device=self.device),
                        }
                        
                        # 将数据移动到设备
                        x_d = x_d.unsqueeze(0).to(self.device)
                        label_d = torch.tensor([label_d], device=self.device)
                        
                        with torch.cuda.amp.autocast():
                            # 前向传播 - 返回分类预测
                            pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_e, pred_d = self.model(
                                x_e=None, label_e=None, x_d=x_d, label_d=label_d, 
                                trajectory_data=trajectory_data, use_trajectory_feature=False, 
                                return_classification=True
                            )
                            # 计算损失：分心分类损失（交叉熵损失） + 分心KL散度损失
                            _, traj_loss, kl_loss_e, kl_loss_d, orth_loss, cls_loss_e, cls_loss_d = self.loss_fn(
                                pred_dict['y_hat'], None, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, 
                                pred_dict['pi'], pred_d=pred_d, label_d=label_d
                            )
                            val_loss += (cls_loss_d + kl_loss_d).item()
                            val_kl_loss_d += kl_loss_d.item()
                
                val_loss_avg = val_loss / (len(val_emotion_dataset) + len(val_distraction_dataset))
                train_loss_avg = total_loss / (len(emotion_train_loader) + len(distraction_train_loader))
                avg_kl_loss_e = total_kl_loss_e / len(emotion_train_loader)
                avg_kl_loss_d = total_kl_loss_d / len(distraction_train_loader)
                
                print(f"Epoch {epoch+1}, Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, KL Loss E: {avg_kl_loss_e:.4f}, KL Loss D: {avg_kl_loss_d:.4f}")
                # 记录损失
                self.loss_history['stage1']['train_loss'].append(train_loss_avg)
                self.loss_history['stage1']['val_loss'].append(val_loss_avg)
            
            # 学习率调度
            distraction_scheduler.step()
            emotion_scheduler.step()
            
            # 保存模型
            if (epoch + 1) % self.config['other_config']['save_interval'] == 0:
                self.save_model(f"stage1_epoch{epoch+1}.pth")
        
        # 训练完成后可视化损失
        self.visualize_loss('stage1')
    
    def train_stage2(self):
        """
        Stage2: FiLM调制训练
        只使用emp数据集，冻结情绪和分心模型
        """
        print("开始 Stage2: FiLM调制训练")
        
        # 获取模型（考虑DataParallel的情况）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 解冻FiLM模块
        for param in model.film_module.parameters():
            param.requires_grad = True
        
        # 保持其他模块冻结状态
        for param in model.emotion_encoder.parameters():
            param.requires_grad = False
        
        for param in model.distraction_encoder.parameters():
            param.requires_grad = False
        
        for param in model.trajectory_encoder.parameters():
            param.requires_grad = True
        
        for param in model.trajectory_decoder.parameters():
            param.requires_grad = True
        
        # 设置学习率调度器
        total_epochs = self.config['train_config']['epochs']['stage2']
        
        # 余弦退火调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs,
            eta_min=self.config['train_config']['learning_rate'] * 0.01  # 最小学习率
        )
        
        # 加载emp数据集（只使用轨迹数据）
        from datasets.trajectory_dataset import TrajectoryDataset
        emp_dataset_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/emp"
        
        # 训练集
        train_emp_dataset = TrajectoryDataset(emp_dataset_dir, split="train", seq_len=50, future_steps=60)
        # 验证集
        val_emp_dataset = TrajectoryDataset(emp_dataset_dir, split="val", seq_len=50, future_steps=60)
        # 测试集
        test_emp_dataset = TrajectoryDataset(emp_dataset_dir, split="test", seq_len=50, future_steps=60)
        
        print(f"Stage2 - emp训练集: {len(train_emp_dataset)}, 验证集: {len(val_emp_dataset)}, 测试集: {len(test_emp_dataset)}")
        
        # 创建数据加载器
        from torch.utils.data import DataLoader
        
        emp_train_loader = DataLoader(
            train_emp_dataset,
            batch_size=self.config['train_config']['batch_size'],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        emp_val_loader = DataLoader(
            val_emp_dataset,
            batch_size=self.config['train_config']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        # 训练FiLM模块和轨迹编码器/解码器
        for epoch in range(self.config['train_config']['epochs']['stage2']):
            self.model.train()
            total_loss = 0
            
            for i, (history, future) in enumerate(tqdm(emp_train_loader, desc=f"Stage2 Epoch {epoch+1}/{self.config['train_config']['epochs']['stage2']}")):
                # 获取批次大小
                batch_size = future.shape[0]
                
                # 构建轨迹数据字典
                num_agents = 1
                trajectory_data = {
                    "x": history.unsqueeze(1).to(self.device),  # [batch_size, num_agents, history_steps, 2]
                    "x_padding_mask": torch.zeros(batch_size, num_agents, history.shape[1], dtype=torch.bool, device=self.device),
                    "x_key_padding_mask": torch.zeros(batch_size, num_agents, dtype=torch.bool, device=self.device),
                    "x_velocity_diff": torch.zeros(batch_size, num_agents, history.shape[1], device=self.device),
                    "x_centers": torch.zeros(batch_size, num_agents, 2, device=self.device),
                    "x_angles": torch.zeros(batch_size, num_agents, history.shape[1] + 1, device=self.device),
                    "x_attr": torch.zeros(batch_size, num_agents, 3, device=self.device, dtype=torch.int),
                    "lane_positions": torch.zeros(batch_size, 5, 20, 2, device=self.device),
                    "lane_centers": torch.zeros(batch_size, 5, 2, device=self.device),
                    "lane_angles": torch.zeros(batch_size, 5, device=self.device),
                    "lane_padding_mask": torch.zeros(batch_size, 5, 20, dtype=torch.bool, device=self.device),
                    "lane_key_padding_mask": torch.zeros(batch_size, 5, dtype=torch.bool, device=self.device),
                }
                
                # 将数据移动到设备
                y = future.to(self.device)
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    # 前向传播 - 使用轨迹特征作为情绪和分心编码器的输入
                    pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(
                        trajectory_data=trajectory_data, use_trajectory_feature=True
                    )
                    
                    # 计算损失（主要是轨迹损失）
                    total_loss_batch, traj_loss, kl_loss_e, kl_loss_d, _ = self.loss_fn(pred_dict['y_hat'], y, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_dict['pi'])
                    loss = traj_loss
                
                # 混合精度反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
            
            # 验证
            if (epoch + 1) % self.config['other_config']['eval_interval'] == 0:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for history, future in tqdm(emp_val_loader, desc="Evaluating Stage2", leave=False):
                        # 获取批次大小
                        batch_size = future.shape[0]
                        
                        # 构建轨迹数据字典
                        num_agents = 1
                        trajectory_data = {
                            "x": history.unsqueeze(1).to(self.device),  # [batch_size, num_agents, history_steps, 2]
                            "x_padding_mask": torch.zeros(batch_size, num_agents, history.shape[1], dtype=torch.bool, device=self.device),
                            "x_key_padding_mask": torch.zeros(batch_size, num_agents, dtype=torch.bool, device=self.device),
                            "x_velocity_diff": torch.zeros(batch_size, num_agents, history.shape[1], device=self.device),
                            "x_centers": torch.zeros(batch_size, num_agents, 2, device=self.device),
                            "x_angles": torch.zeros(batch_size, num_agents, history.shape[1] + 1, device=self.device),
                            "x_attr": torch.zeros(batch_size, num_agents, 3, device=self.device, dtype=torch.int),
                            "lane_positions": torch.zeros(batch_size, 5, 20, 2, device=self.device),
                            "lane_centers": torch.zeros(batch_size, 5, 2, device=self.device),
                            "lane_angles": torch.zeros(batch_size, 5, device=self.device),
                            "lane_padding_mask": torch.zeros(batch_size, 5, 20, dtype=torch.bool, device=self.device),
                            "lane_key_padding_mask": torch.zeros(batch_size, 5, dtype=torch.bool, device=self.device),
                        }
                        
                        # 将数据移动到设备
                        y = future.to(self.device)
                        
                        # 前向传播 - 使用轨迹特征作为情绪和分心编码器的输入
                        pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(
                            trajectory_data=trajectory_data, use_trajectory_feature=True
                        )
                        
                        # 计算损失（主要是轨迹损失）
                        total_loss_batch, traj_loss, kl_loss_e, kl_loss_d, _ = self.loss_fn(pred_dict['y_hat'], y, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_dict['pi'])
                        val_loss += traj_loss.item()
                
                val_loss_avg = val_loss / len(emp_val_loader)
                train_loss_avg = total_loss / len(emp_train_loader)
                print(f"Epoch {epoch+1}, Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")
                # 记录损失
                self.loss_history['stage2']['train_loss'].append(train_loss_avg)
                self.loss_history['stage2']['val_loss'].append(val_loss_avg)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 保存模型
            if (epoch + 1) % self.config['other_config']['save_interval'] == 0:
                self.save_model(f"stage2_epoch{epoch+1}.pth")
        
        # 训练完成后可视化损失
        self.visualize_loss('stage2')
    
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
                total_loss_batch, _, _, _, _ = self.loss_fn(pred_dict['y_hat'], y, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_dict['pi'])
                total_loss += total_loss_batch.item()
        
        return total_loss / len(self.val_loader)
    
    def save_model(self, filename):
        """
        保存模型
        
        Args:
            filename: 文件名
        """
        # 获取原始模型（考虑DataParallel的情况）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        save_path = os.path.join(self.config['other_config']['checkpoint_dir'], filename)
        torch.save(model.state_dict(), save_path)
        print(f"模型保存到 {save_path}")
    
    def load_model(self, filename):
        """
        加载模型
        
        Args:
            filename: 文件名
        """
        # 获取原始模型（考虑DataParallel的情况）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        load_path = os.path.join(self.config['other_config']['checkpoint_dir'], filename)
        model.load_state_dict(torch.load(load_path))
        print(f"从 {load_path} 加载模型")
    
    def visualize_loss(self, stage):
        """
        可视化损失曲线
        
        Args:
            stage: 训练阶段名称
        """
        # 创建logs目录
        os.makedirs(self.config['other_config']['log_dir'], exist_ok=True)
        
        # 获取损失数据
        train_loss = self.loss_history[stage]['train_loss']
        val_loss = self.loss_history[stage]['val_loss']
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{stage} Loss Curve')
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        save_path = os.path.join(self.config['other_config']['log_dir'], f'{stage}_loss_curve.png')
        plt.savefig(save_path)
        print(f"损失曲线保存到 {save_path}")
        plt.close()

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
    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2], help='训练阶段')
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
        trainer.load_model(args.resume)
    
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
    except KeyboardInterrupt:
        print("训练被用户中断")
    finally:
        # 清理资源
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

