import os
import time
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from .data_loader import (
    get_trajectory_loader,
    get_emotion_loader,
    get_distraction_loader,
    move_trajectory_data_to_device,
    create_minimal_trajectory_data_dict
)


class Stage0Trainer:
    """
    Stage0: 轨迹网络预训练
    """
    def __init__(self, model, device, loss_fn, optimizer, scheduler, config):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    def load_pretrained_weights(self):
        """
        加载情绪和分心模型的预训练权重
        """
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
    
    def freeze_modules(self):
        """
        冻结情绪、分心编码器和FiLM模块
        """
        # 获取模型（考虑DataParallel的情况）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 冻结情绪和分心编码器
        for param in model.emotion_encoder.parameters():
            param.requires_grad = False
        
        for param in model.distraction_encoder.parameters():
            param.requires_grad = False
        
        # 冻结FiLM模块
        for param in model.film_module.parameters():
            param.requires_grad = False
    
    def train(self, loss_history, best_val_loss, save_model_fn, visualize_fn):
        """
        训练轨迹编码器和解码器
        """
        print("开始 Stage0: 轨迹网络预训练")
        
        # 加载预训练权重
        self.load_pretrained_weights()
        
        # 冻结模块
        self.freeze_modules()
        
        # 加载emp数据集
        emp_dataset_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/emp"
        
        # 创建数据加载器
        emp_train_loader = get_trajectory_loader(
            emp_dataset_dir, "train", 
            self.config['train_config']['batch_size']
        )
        
        emp_val_loader = get_trajectory_loader(
            emp_dataset_dir, "val", 
            self.config['train_config']['batch_size']
        )
        
        test_emp_loader = get_trajectory_loader(
            emp_dataset_dir, "test", 
            self.config['train_config']['batch_size']
        )
        
        print(f"Stage0 - emp训练集: {len(emp_train_loader.dataset)}, 验证集: {len(emp_val_loader.dataset)}, 测试集: {len(test_emp_loader.dataset)}")
        
        # 训练轨迹编码器和解码器
        for epoch in range(self.config['train_config']['epochs']['stage0']):
            self.model.train()
            total_loss = 0
            total_reg_loss = 0
            total_cls_loss = 0
            
            for i, data in enumerate(tqdm(emp_train_loader, desc=f"Stage0 Epoch {epoch+1}/{self.config['train_config']['epochs']['stage0']}")):
                # 记录批次开始时间
                start_time = time.time()
                
                # 将数据移动到设备
                data = move_trajectory_data_to_device(data, self.device)
                
                try:
                    # 混合精度前向传播
                    with torch.cuda.amp.autocast():
                        # 前向传播 - 使用轨迹特征作为情绪和分心编码器的输入
                        pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(
                            trajectory_data=data, use_trajectory_feature=True
                        )
                        
                        # 计算损失（参考trainer_forecast.py的实现）
                        y_hat, pi = pred_dict['y_hat'], pred_dict['pi']
                        y = data["y"][:, 0]  # 目标车辆的真实轨迹
                        
                        B = y_hat.shape[0]
                        B_range = range(B)
                        
                        # 计算L2范数
                        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)
                        
                        # 找到最佳预测模式
                        best_mode = torch.argmin(l2_norm, dim=-1)
                        y_hat_best = y_hat[B_range, best_mode]
                        
                        # 计算回归损失
                        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
                        
                        # 计算分类损失
                        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
                        
                        # 总损失
                        loss = agent_reg_loss + agent_cls_loss
                    
                    # 混合精度反向传播
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    total_loss += loss.item()
                    total_reg_loss += agent_reg_loss.item()
                    total_cls_loss += agent_cls_loss.item()
                    
                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # 验证
            if (epoch + 1) % self.config['other_config']['eval_interval'] == 0:
                self.model.eval()
                val_loss = 0
                val_reg_loss = 0
                val_cls_loss = 0
                val_min_ade = 0
                val_min_fde = 0
                val_mr = 0
                with torch.no_grad():
                    for data in tqdm(emp_val_loader, desc="Evaluating Stage0", leave=False):
                        # 将数据移动到设备
                        data = move_trajectory_data_to_device(data, self.device)
                        
                        # 前向传播 - 使用轨迹特征作为情绪和分心编码器的输入
                        pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(
                            trajectory_data=data, use_trajectory_feature=True
                        )
                        
                        # 计算损失（参考trainer_forecast.py的实现）
                        y_hat, pi = pred_dict['y_hat'], pred_dict['pi']
                        y = data["y"][:, 0]  # 目标车辆的真实轨迹
                        
                        B = y_hat.shape[0]
                        B_range = range(B)
                        
                        # 计算L2范数
                        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)
                        
                        # 找到最佳预测模式
                        best_mode = torch.argmin(l2_norm, dim=-1)
                        y_hat_best = y_hat[B_range, best_mode]
                        
                        # 计算回归损失
                        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
                        
                        # 计算分类损失
                        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
                        
                        # 总损失
                        loss = agent_reg_loss + agent_cls_loss
                        
                        # 计算评价指标
                        # minADE
                        ade = torch.norm(y_hat[..., :2] - y.unsqueeze(1), p=2, dim=-1).mean(-1)
                        min_ade = ade.min(-1)[0].mean()
                        
                        # minFDE
                        fde = torch.norm(y_hat[..., -1, :2] - y.unsqueeze(1)[..., -1, :2], p=2, dim=-1)
                        min_fde = fde.min(-1)[0].mean()
                        
                        # MR
                        miss_threshold = 2.0
                        missed_pred = fde > miss_threshold
                        mr = missed_pred.all(-1).float().mean()
                        
                        val_loss += loss.item()
                        val_reg_loss += agent_reg_loss.item()
                        val_cls_loss += agent_cls_loss.item()
                        val_min_ade += min_ade.item()
                        val_min_fde += min_fde.item()
                        val_mr += mr.item()
                
                val_loss_avg = val_loss / len(emp_val_loader)
                val_reg_loss_avg = val_reg_loss / len(emp_val_loader)
                val_cls_loss_avg = val_cls_loss / len(emp_val_loader)
                val_min_ade_avg = val_min_ade / len(emp_val_loader)
                val_min_fde_avg = val_min_fde / len(emp_val_loader)
                val_mr_avg = val_mr / len(emp_val_loader)
                train_loss_avg = total_loss / len(emp_train_loader)
                train_reg_loss_avg = total_reg_loss / len(emp_train_loader)
                train_cls_loss_avg = total_cls_loss / len(emp_train_loader)
                
                print(f"Epoch {epoch+1}, Train Loss: {train_loss_avg:.4f} (Reg: {train_reg_loss_avg:.4f}, Cls: {train_cls_loss_avg:.4f})")
                print(f"Epoch {epoch+1}, Val Loss: {val_loss_avg:.4f} (Reg: {val_reg_loss_avg:.4f}, Cls: {val_cls_loss_avg:.4f})")
                print(f"Epoch {epoch+1}, Val minADE: {val_min_ade_avg:.4f}, Val minFDE: {val_min_fde_avg:.4f}, Val MR: {val_mr_avg:.4f}")
                
                # 记录损失和评价指标
                loss_history['stage0']['train_loss'].append(train_loss_avg)
                loss_history['stage0']['val_loss'].append(val_loss_avg)
                # 记录评价指标
                if 'val_minADE' not in loss_history['stage0']:
                    loss_history['stage0']['val_minADE'] = []
                    loss_history['stage0']['val_minFDE'] = []
                    loss_history['stage0']['val_MR'] = []
                loss_history['stage0']['val_minADE'].append(val_min_ade_avg)
                loss_history['stage0']['val_minFDE'].append(val_min_fde_avg)
                loss_history['stage0']['val_MR'].append(val_mr_avg)
                
                # 写入日志
                log_path = os.path.join(self.config['other_config']['log_dir'], f'stage0_training.log')
                with open(log_path, 'a') as f:
                    f.write(f"Epoch {epoch+1}, Train Loss: {train_loss_avg:.4f} (Reg: {train_reg_loss_avg:.4f}, Cls: {train_cls_loss_avg:.4f})\n")
                    f.write(f"Epoch {epoch+1}, Val Loss: {val_loss_avg:.4f} (Reg: {val_reg_loss_avg:.4f}, Cls: {val_cls_loss_avg:.4f})\n")
                    f.write(f"Epoch {epoch+1}, Val minADE: {val_min_ade_avg:.4f}, Val minFDE: {val_min_fde_avg:.4f}, Val MR: {val_mr_avg:.4f}\n")
                    f.write("\n")
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 保存最新模型
            save_model_fn("stage0_latest.pth")
            
            # 保存最佳模型
            if val_loss_avg < best_val_loss['stage0']:
                best_val_loss['stage0'] = val_loss_avg
                save_model_fn("stage0_best.pth")
                print(f"保存最佳模型，验证损失: {val_loss_avg:.4f}")
        
        # 训练完成后可视化损失
        visualize_fn('stage0')


class Stage1Trainer:
    """
    Stage1: 情绪和分心编码器训练
    """
    def __init__(self, model, device, loss_fn, config):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.config = config
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.emotion_optimizer = None
        self.distraction_optimizer = None
        self.emotion_scheduler = None
        self.distraction_scheduler = None
    
    def setup_optimizers(self):
        """
        设置优化器和学习率调度器
        """
        # 获取模型（考虑DataParallel的情况）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 为分心模型创建单独的优化器
        self.distraction_optimizer = optim.AdamW(
            model.distraction_encoder.parameters(),
            lr=self.config['train_config']['learning_rate'] * 1.5,  # 分心模型学习率稍高
            weight_decay=self.config['train_config']['weight_decay']
        )
        
        # 为情绪模型创建优化器
        self.emotion_optimizer = optim.AdamW(
            model.emotion_encoder.parameters(),
            lr=self.config['train_config']['learning_rate']*1.5,
            weight_decay=self.config['train_config']['weight_decay']
        )
        
        # 设置学习率调度器
        total_epochs = self.config['train_config']['epochs']['stage1']
        warmup_epochs = 5
        
        # 余弦退火调度器，带预热
        self.distraction_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.distraction_optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=self.config['train_config']['learning_rate'] * 0.01  # 最小学习率
        )
        
        self.emotion_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.emotion_optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=self.config['train_config']['learning_rate'] * 0.01  # 最小学习率
        )
        
        # 预热调度器
        if warmup_epochs > 0:
            warmup_distraction_scheduler = optim.lr_scheduler.LinearLR(
                self.distraction_optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            self.distraction_scheduler = optim.lr_scheduler.SequentialLR(
                self.distraction_optimizer,
                schedulers=[warmup_distraction_scheduler, self.distraction_scheduler],
                milestones=[warmup_epochs]
            )
            
            warmup_emotion_scheduler = optim.lr_scheduler.LinearLR(
                self.emotion_optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            self.emotion_scheduler = optim.lr_scheduler.SequentialLR(
                self.emotion_optimizer,
                schedulers=[warmup_emotion_scheduler, self.emotion_scheduler],
                milestones=[warmup_epochs]
            )
    
    def unfreeze_modules(self):
        """
        解冻情绪和分心编码器，冻结轨迹编码器和解码器
        """
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
    
    def train(self, loss_history, best_val_loss, save_model_fn, load_model_fn, visualize_fn):
        """
        训练情绪和分心编码器
        """
        print("开始 Stage1: 情绪和分心编码器训练")
        
        # 解冻模块
        self.unfreeze_modules()
        
        # 设置优化器
        self.setup_optimizers()
        
        # 加载情绪和分心数据集
        emotion_dataset_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/Constructed_Small_sample_0.85"
        distraction_dataset_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/SFDDD/images"
        
        # 创建数据加载器
        emotion_train_loader = get_emotion_loader(
            emotion_dataset_dir, "train", 
            self.config['train_config']['batch_size']
        )
        
        distraction_train_loader = get_distraction_loader(
            distraction_dataset_dir, "train", 
            self.config['train_config']['batch_size']
        )
        
        val_emotion_loader = get_emotion_loader(
            emotion_dataset_dir, "val", 
            self.config['train_config']['batch_size']
        )
        
        val_distraction_loader = get_distraction_loader(
            distraction_dataset_dir, "val", 
            self.config['train_config']['batch_size']
        )
        
        print(f"Stage1 - 情绪训练集: {len(emotion_train_loader.dataset)}, 验证集: {len(val_emotion_loader.dataset)}")
        print(f"Stage1 - 分心训练集: {len(distraction_train_loader.dataset)}, 验证集: {len(val_distraction_loader.dataset)}")
        
        # 训练情绪和分心编码器
        for epoch in range(self.config['train_config']['epochs']['stage1']):
            self.model.train()
            total_loss_e = 0
            total_loss_d = 0
            total_kl_loss_e = 0
            total_kl_loss_d = 0
            
            # 训练情绪编码器
            print(f"训练情绪编码器...")
            for i, (x_e, label_e) in enumerate(tqdm(emotion_train_loader, desc=f"Stage1 Epoch {epoch+1}/{self.config['train_config']['epochs']['stage1']} - Emotion")):
                # 构建最小化的轨迹数据字典（仅用于模型调用）
                batch_size = x_e.shape[0]
                trajectory_data = create_minimal_trajectory_data_dict(batch_size, self.device)
                
                # 将数据移动到设备
                x_e = x_e.to(self.device)
                label_e = label_e.to(self.device)
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    # 前向传播 - 只训练情绪模型，返回分类预测
                    pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_e, pred_d = self.model(
                        x_e=x_e, label_e=label_e, x_d=None, label_d=None, 
                        trajectory_data=trajectory_data, use_trajectory_feature=False, 
                        return_classification=True
                    )
                    
                    # 计算损失：情绪分类损失（Focal Loss） + 情绪KL散度损失
                    _, traj_loss, kl_loss_e, kl_loss_d, orth_loss, cls_loss_e, cls_loss_d, _, _, _ = self.loss_fn(
                        pred_dict['y_hat'], None, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, 
                        pred_dict['pi'], pred_e=pred_e, label_e=label_e
                    )
                    loss = cls_loss_e + 0.8*kl_loss_e
                
                # 混合精度反向传播
                self.emotion_optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.emotion_optimizer)
                self.scaler.update()
                
                total_loss_e += loss.item()
                total_kl_loss_e += kl_loss_e.item()
            
            # 训练分心编码器
            print(f"训练分心编码器...")
            for i, (x_d, label_d) in enumerate(tqdm(distraction_train_loader, desc=f"Stage1 Epoch {epoch+1}/{self.config['train_config']['epochs']['stage1']} - Distraction")):
                # 构建最小化的轨迹数据字典（仅用于模型调用）
                batch_size = x_d.shape[0]
                trajectory_data = create_minimal_trajectory_data_dict(batch_size, self.device)
                
                # 将数据移动到设备
                x_d = x_d.to(self.device)
                label_d = label_d.to(self.device)
                
                # 检查并裁剪标签范围
                # 获取模型（考虑DataParallel的情况）
                model = self.model.module if hasattr(self.model, 'module') else self.model
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
                    _, traj_loss, kl_loss_e, kl_loss_d, orth_loss, cls_loss_e, cls_loss_d, _, _, _ = self.loss_fn(
                        pred_dict['y_hat'], None, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, 
                        pred_dict['pi'], pred_d=pred_d, label_d=label_d
                    )
                    loss = cls_loss_d + kl_loss_d
                
                # 混合精度反向传播
                self.distraction_optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.distraction_optimizer)
                self.scaler.update()
                
                total_loss_d += loss.item()
                total_kl_loss_d += kl_loss_d.item()
            
            # 验证
            if (epoch + 1) % self.config['other_config']['eval_interval'] == 0:
                val_loss_e = 0
                val_loss_d = 0
                val_kl_loss_e = 0
                val_kl_loss_d = 0
                val_count_e = 0
                val_count_d = 0
                
                # 初始化混淆矩阵数据
                if 'emotion_true_labels' not in loss_history['stage1']:
                    loss_history['stage1']['emotion_true_labels'] = []
                    loss_history['stage1']['emotion_pred_labels'] = []
                    loss_history['stage1']['distraction_true_labels'] = []
                    loss_history['stage1']['distraction_pred_labels'] = []
                
                self.model.eval()
                with torch.no_grad():
                    # 验证情绪编码器
                    for x_e, label_e in val_emotion_loader:
                        # 构建最小化的轨迹数据字典（仅用于模型调用）
                        batch_size = x_e.shape[0]
                        trajectory_data = create_minimal_trajectory_data_dict(batch_size, self.device)
                        
                        # 将数据移动到设备
                        x_e = x_e.to(self.device)
                        label_e = label_e.to(self.device)
                        
                        with torch.cuda.amp.autocast():
                            # 前向传播 - 返回分类预测
                            pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_e, pred_d = self.model(
                                x_e=x_e, label_e=label_e, x_d=None, label_d=None, 
                                trajectory_data=trajectory_data, use_trajectory_feature=False, 
                                return_classification=True
                            )
                            # 计算损失：情绪分类损失（交叉熵损失） + 情绪KL散度损失
                            _, traj_loss, kl_loss_e, kl_loss_d, orth_loss, cls_loss_e, cls_loss_d, _, _, _ = self.loss_fn(
                                pred_dict['y_hat'], None, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, 
                                pred_dict['pi'], pred_e=pred_e, label_e=label_e
                            )
                            val_loss_e += (cls_loss_e + kl_loss_e).item() * batch_size
                            val_kl_loss_e += kl_loss_e.item() * batch_size
                            val_count_e += batch_size
                            
                            # 记录混淆矩阵数据
                            pred_e_labels = torch.argmax(pred_e, dim=1)
                            loss_history['stage1']['emotion_true_labels'].extend(label_e.cpu().numpy().tolist())
                            loss_history['stage1']['emotion_pred_labels'].extend(pred_e_labels.cpu().numpy().tolist())
                    
                    # 验证分心编码器
                    for x_d, label_d in val_distraction_loader:
                        # 构建最小化的轨迹数据字典（仅用于模型调用）
                        batch_size = x_d.shape[0]
                        trajectory_data = create_minimal_trajectory_data_dict(batch_size, self.device)
                        
                        # 将数据移动到设备
                        x_d = x_d.to(self.device)
                        label_d = label_d.to(self.device)
                        
                        with torch.cuda.amp.autocast():
                            # 前向传播 - 返回分类预测
                            pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_e, pred_d = self.model(
                                x_e=None, label_e=None, x_d=x_d, label_d=label_d, 
                                trajectory_data=trajectory_data, use_trajectory_feature=False, 
                                return_classification=True
                            )
                            # 计算损失：分心分类损失（交叉熵损失） + 分心KL散度损失
                            _, traj_loss, kl_loss_e, kl_loss_d, orth_loss, cls_loss_e, cls_loss_d, _, _, _ = self.loss_fn(
                                pred_dict['y_hat'], None, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, 
                                pred_dict['pi'], pred_d=pred_d, label_d=label_d
                            )
                            val_loss_d += (cls_loss_d + kl_loss_d).item() * batch_size
                            val_kl_loss_d += kl_loss_d.item() * batch_size
                            val_count_d += batch_size
                            
                            # 记录混淆矩阵数据
                            pred_d_labels = torch.argmax(pred_d, dim=1)
                            loss_history['stage1']['distraction_true_labels'].extend(label_d.cpu().numpy().tolist())
                            loss_history['stage1']['distraction_pred_labels'].extend(pred_d_labels.cpu().numpy().tolist())
                
                # 分别计算情绪和分心的验证损失
                val_loss_e_avg = val_loss_e / val_count_e if val_count_e > 0 else 0
                val_loss_d_avg = val_loss_d / val_count_d if val_count_d > 0 else 0
                
                # 计算训练损失
                train_loss_e_avg = total_loss_e / len(emotion_train_loader) if len(emotion_train_loader) > 0 else 0
                train_loss_d_avg = total_loss_d / len(distraction_train_loader) if len(distraction_train_loader) > 0 else 0
                avg_kl_loss_e = total_kl_loss_e / len(emotion_train_loader) if len(emotion_train_loader) > 0 else 0
                avg_kl_loss_d = total_kl_loss_d / len(distraction_train_loader) if len(distraction_train_loader) > 0 else 0
                
                # 计算评估指标
                # 计算情绪分类指标
                emotion_acc = 0
                emotion_f1 = 0
                if len(loss_history['stage1']['emotion_true_labels']) > 0:
                    emotion_acc = accuracy_score(loss_history['stage1']['emotion_true_labels'], loss_history['stage1']['emotion_pred_labels'])
                    emotion_f1 = f1_score(loss_history['stage1']['emotion_true_labels'], loss_history['stage1']['emotion_pred_labels'], average='weighted')
                
                # 计算分心分类指标
                distraction_acc = 0
                distraction_f1 = 0
                if len(loss_history['stage1']['distraction_true_labels']) > 0:
                    distraction_acc = accuracy_score(loss_history['stage1']['distraction_true_labels'], loss_history['stage1']['distraction_pred_labels'])
                    distraction_f1 = f1_score(loss_history['stage1']['distraction_true_labels'], loss_history['stage1']['distraction_pred_labels'], average='weighted')
                
                # 打印损失信息和评估指标
                print(f"Epoch {epoch+1}, Emotion Train Loss: {train_loss_e_avg:.4f}, Emotion Val Loss: {val_loss_e_avg:.4f}")
                print(f"Epoch {epoch+1}, Emotion Acc: {emotion_acc:.4f}, Emotion F1: {emotion_f1:.4f}")
                print(f"Epoch {epoch+1}, Distraction Train Loss: {train_loss_d_avg:.4f}, Distraction Val Loss: {val_loss_d_avg:.4f}")
                print(f"Epoch {epoch+1}, Distraction Acc: {distraction_acc:.4f}, Distraction F1: {distraction_f1:.4f}")
                print(f"Epoch {epoch+1}, KL Loss E: {avg_kl_loss_e:.4f}, KL Loss D: {avg_kl_loss_d:.4f}")
                
                # 写入日志
                log_path = os.path.join(self.config['other_config']['log_dir'], f'stage1_training.log')
                with open(log_path, 'a') as f:
                    f.write(f"Epoch {epoch+1}, Emotion Train Loss: {train_loss_e_avg:.4f}, Emotion Val Loss: {val_loss_e_avg:.4f}\n")
                    f.write(f"Epoch {epoch+1}, Emotion Acc: {emotion_acc:.4f}, Emotion F1: {emotion_f1:.4f}\n")
                    f.write(f"Epoch {epoch+1}, Distraction Train Loss: {train_loss_d_avg:.4f}, Distraction Val Loss: {val_loss_d_avg:.4f}\n")
                    f.write(f"Epoch {epoch+1}, Distraction Acc: {distraction_acc:.4f}, Distraction F1: {distraction_f1:.4f}\n")
                    f.write(f"Epoch {epoch+1}, KL Loss E: {avg_kl_loss_e:.4f}, KL Loss D: {avg_kl_loss_d:.4f}\n")
                    f.write("\n")
                
                # 记录损失和评估指标
                if 'train_loss_e' not in loss_history['stage1']:
                    loss_history['stage1']['train_loss_e'] = []
                    loss_history['stage1']['val_loss_e'] = []
                    loss_history['stage1']['train_loss_d'] = []
                    loss_history['stage1']['val_loss_d'] = []
                    loss_history['stage1']['emotion_true_labels'] = []
                    loss_history['stage1']['emotion_pred_labels'] = []
                    loss_history['stage1']['distraction_true_labels'] = []
                    loss_history['stage1']['distraction_pred_labels'] = []
                    loss_history['stage1']['emotion_acc'] = []
                    loss_history['stage1']['emotion_f1'] = []
                    loss_history['stage1']['distraction_acc'] = []
                    loss_history['stage1']['distraction_f1'] = []
                
                loss_history['stage1']['train_loss_e'].append(train_loss_e_avg)
                loss_history['stage1']['val_loss_e'].append(val_loss_e_avg)
                loss_history['stage1']['train_loss_d'].append(train_loss_d_avg)
                loss_history['stage1']['val_loss_d'].append(val_loss_d_avg)
                loss_history['stage1']['emotion_acc'].append(emotion_acc)
                loss_history['stage1']['emotion_f1'].append(emotion_f1)
                loss_history['stage1']['distraction_acc'].append(distraction_acc)
                loss_history['stage1']['distraction_f1'].append(distraction_f1)
            
            # 学习率调度
            self.distraction_scheduler.step()
            self.emotion_scheduler.step()
            
            # 保存最新模型
            save_model_fn("stage1_latest.pth")
            
            # 保存最佳模型
            if val_loss_e_avg < best_val_loss.get('stage1_emotion', float('inf')):
                best_val_loss['stage1_emotion'] = val_loss_e_avg
                save_model_fn("stage1_best_emotion.pth")
                print(f"保存最佳情绪模型，验证损失: {val_loss_e_avg:.4f}")
            
            if val_loss_d_avg < best_val_loss.get('stage1_distraction', float('inf')):
                best_val_loss['stage1_distraction'] = val_loss_d_avg
                save_model_fn("stage1_best_distraction.pth")
                print(f"保存最佳分心模型，验证损失: {val_loss_d_avg:.4f}")
        
        # 训练完成后，加载最佳模型并绘制混淆矩阵
        print("加载最佳模型并生成混淆矩阵...")
        
        # 加载最佳情绪模型
        best_emotion_model_path = os.path.join(self.config['other_config']['checkpoint_dir'], "stage1_best_emotion.pth")
        if os.path.exists(best_emotion_model_path):
            load_model_fn("stage1_best_emotion.pth")
            print("加载最佳情绪模型成功")
        else:
            print("最佳情绪模型文件不存在，使用当前模型")
        
        # 清空之前的混淆矩阵数据
        loss_history['stage1']['emotion_true_labels'] = []
        loss_history['stage1']['emotion_pred_labels'] = []
        loss_history['stage1']['distraction_true_labels'] = []
        loss_history['stage1']['distraction_pred_labels'] = []
        
        # 重新验证，收集混淆矩阵数据
        self.model.eval()
        with torch.no_grad():
            # 验证情绪编码器
            for x_e, label_e in val_emotion_loader:
                # 构建最小化的轨迹数据字典（仅用于模型调用）
                batch_size = x_e.shape[0]
                trajectory_data = create_minimal_trajectory_data_dict(batch_size, self.device)
                
                # 将数据移动到设备
                x_e = x_e.to(self.device)
                label_e = label_e.to(self.device)
                
                with torch.cuda.amp.autocast():
                    # 前向传播 - 返回分类预测
                    pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_e, pred_d = self.model(
                        x_e=x_e, label_e=label_e, x_d=None, label_d=None, 
                        trajectory_data=trajectory_data, use_trajectory_feature=False, 
                        return_classification=True
                    )
                    
                    # 记录混淆矩阵数据
                    pred_e_labels = torch.argmax(pred_e, dim=1)
                    loss_history['stage1']['emotion_true_labels'].extend(label_e.cpu().numpy().tolist())
                    loss_history['stage1']['emotion_pred_labels'].extend(pred_e_labels.cpu().numpy().tolist())
            
            # 加载最佳分心模型
            best_distraction_model_path = os.path.join(self.config['other_config']['checkpoint_dir'], "stage1_best_distraction.pth")
            if os.path.exists(best_distraction_model_path):
                load_model_fn("stage1_best_distraction.pth")
                print("加载最佳分心模型成功")
            else:
                print("最佳分心模型文件不存在，使用当前模型")
            
            # 验证分心编码器
            for x_d, label_d in val_distraction_loader:
                # 构建最小化的轨迹数据字典（仅用于模型调用）
                batch_size = x_d.shape[0]
                trajectory_data = create_minimal_trajectory_data_dict(batch_size, self.device)
                
                # 将数据移动到设备
                x_d = x_d.to(self.device)
                label_d = label_d.to(self.device)
                
                with torch.cuda.amp.autocast():
                    # 前向传播 - 返回分类预测
                    pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d, pred_e, pred_d = self.model(
                        x_e=None, label_e=None, x_d=x_d, label_d=label_d, 
                        trajectory_data=trajectory_data, use_trajectory_feature=False, 
                        return_classification=True
                    )
                    
                    # 记录混淆矩阵数据
                    pred_d_labels = torch.argmax(pred_d, dim=1)
                    loss_history['stage1']['distraction_true_labels'].extend(label_d.cpu().numpy().tolist())
                    loss_history['stage1']['distraction_pred_labels'].extend(pred_d_labels.cpu().numpy().tolist())
        
        # 训练完成后可视化损失和混淆矩阵
        visualize_fn('stage1')
        
        # 创建联合最佳模型
        print("创建联合最佳模型...")
        
        # 获取模型实例
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 1. 加载最佳情绪模型
        best_emotion_model_path = os.path.join(self.config['other_config']['checkpoint_dir'], "stage1_best_emotion.pth")
        if os.path.exists(best_emotion_model_path):
            # 加载完整的情绪模型权重
            emotion_state_dict = torch.load(best_emotion_model_path)
            model.load_state_dict(emotion_state_dict, strict=False)
            print("加载最佳情绪模型成功")
        
        # 2. 加载最佳分心模型的权重（只加载分心编码器部分）
        best_distraction_model_path = os.path.join(self.config['other_config']['checkpoint_dir'], "stage1_best_distraction.pth")
        if os.path.exists(best_distraction_model_path):
            # 加载分心模型权重
            distraction_state_dict = torch.load(best_distraction_model_path)
            # 过滤出只属于分心编码器的权重
            distraction_encoder_state_dict = {
                k: v for k, v in distraction_state_dict.items() 
                if k.startswith('distraction_encoder.')
            }
            # 只加载分心编码器的权重，避免覆盖其他部分
            model.load_state_dict(distraction_encoder_state_dict, strict=False)
            print("加载最佳分心编码器权重成功")
        
        # 3. 保存联合最佳模型
        save_model_fn("stage1_best_combined.pth")
        print("联合最佳模型保存成功: stage1_best_combined.pth")


class Stage3Trainer:
    """
    Stage3: 联合微调训练
    加载轨迹模型、情绪模型、分心模型、FiLM权重，以更小的学习率完成整个流程的权重微调
    """
    def __init__(self, model, device, loss_fn, config):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.config = config
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.optimizer = None
        self.scheduler = None
    
    def setup_optimizer(self):
        """
        设置优化器和学习率调度器（使用更小的学习率）
        """
        # 使用更小的学习率进行微调
        learning_rate = self.config['train_config']['learning_rate'] * 0.1
        
        # 为整个模型创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config['train_config']['weight_decay']
        )
        
        # 设置学习率调度器
        total_epochs = self.config['train_config']['epochs'].get('stage3', 10)
        
        # 余弦退火调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs,
            eta_min=learning_rate * 0.01  # 最小学习率
        )
    
    def setup_modules(self):
        """
        设置模块的训练状态（全部解冻，进行微调）
        """
        # 获取模型（考虑DataParallel的情况）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 解冻所有模块，进行微调
        for param in model.parameters():
            param.requires_grad = True
    
    def train(self, loss_history, best_val_loss, save_model_fn, load_model_fn, visualize_fn):
        """
        联合微调训练
        """
        print("开始 Stage3: 联合微调训练")
        
        # 加载Stage2的最佳模型权重
        stage2_best_model_path = os.path.join(self.config['other_config']['checkpoint_dir'], "stage2_best.pth")
        if os.path.exists(stage2_best_model_path):
            load_model_fn("stage2_best.pth")
            print("加载Stage2最佳模型成功")
        else:
            print("Stage2最佳模型文件不存在，使用当前模型")
        
        # 设置模块
        self.setup_modules()
        
        # 设置优化器
        self.setup_optimizer()
        
        # 加载emp数据集
        emp_dataset_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/emp"
        
        # 创建数据加载器
        emp_train_loader = get_trajectory_loader(
            emp_dataset_dir, "train", 
            self.config['train_config']['batch_size']
        )
        
        emp_val_loader = get_trajectory_loader(
            emp_dataset_dir, "val", 
            self.config['train_config']['batch_size'],
            num_workers=4
        )
        
        test_emp_loader = get_trajectory_loader(
            emp_dataset_dir, "test", 
            self.config['train_config']['batch_size']
        )
        
        print(f"Stage3 - emp训练集: {len(emp_train_loader.dataset)}, 验证集: {len(emp_val_loader.dataset)}, 测试集: {len(test_emp_loader.dataset)}")
        
        # 训练所有模块
        total_epochs = self.config['train_config']['epochs'].get('stage3', 10)
        for epoch in range(total_epochs):
            self.model.train()
            total_loss = 0
            total_reg_loss = 0
            total_cls_loss = 0
            
            for i, data in enumerate(tqdm(emp_train_loader, desc=f"Stage3 Epoch {epoch+1}/{total_epochs}")):
                # 将数据移动到设备
                data = move_trajectory_data_to_device(data, self.device)
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    # 前向传播 - 使用轨迹特征作为情绪和分心编码器的输入
                    pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(
                        trajectory_data=data, use_trajectory_feature=True
                    )
                    
                    # 计算损失（参考trainer_forecast.py的实现）
                    y_hat, pi = pred_dict['y_hat'], pred_dict['pi']
                    y = data["y"][:, 0]  # 目标车辆的真实轨迹
                    
                    B = y_hat.shape[0]
                    B_range = range(B)
                    
                    # 计算L2范数
                    l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)
                    
                    # 找到最佳预测模式
                    best_mode = torch.argmin(l2_norm, dim=-1)
                    y_hat_best = y_hat[B_range, best_mode]
                    
                    # 计算回归损失
                    agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
                    
                    # 计算分类损失
                    agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
                    
                    # 总损失
                    loss = agent_reg_loss + agent_cls_loss
                
                # 混合精度反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                total_reg_loss += agent_reg_loss.item()
                total_cls_loss += agent_cls_loss.item()
            
            # 验证
            if (epoch + 1) % self.config['other_config']['eval_interval'] == 0:
                self.model.eval()
                val_loss = 0
                val_reg_loss = 0
                val_cls_loss = 0
                val_min_ade = 0
                val_min_fde = 0
                val_mr = 0
                with torch.no_grad():
                    for data in tqdm(emp_val_loader, desc="Evaluating Stage3", leave=False):
                        # 将数据移动到设备
                        data = move_trajectory_data_to_device(data, self.device)
                        
                        # 前向传播 - 使用轨迹特征作为情绪和分心编码器的输入
                        pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(
                            trajectory_data=data, use_trajectory_feature=True
                        )
                        
                        # 计算损失（参考trainer_forecast.py的实现）
                        y_hat, pi = pred_dict['y_hat'], pred_dict['pi']
                        y = data["y"][:, 0]  # 目标车辆的真实轨迹
                        
                        B = y_hat.shape[0]
                        B_range = range(B)
                        
                        # 计算L2范数
                        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)
                        
                        # 找到最佳预测模式
                        best_mode = torch.argmin(l2_norm, dim=-1)
                        y_hat_best = y_hat[B_range, best_mode]
                        
                        # 计算回归损失
                        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
                        
                        # 计算分类损失
                        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
                        
                        # 总损失
                        loss = agent_reg_loss + agent_cls_loss
                        
                        # 计算评价指标
                        # minADE
                        ade = torch.norm(y_hat[..., :2] - y.unsqueeze(1), p=2, dim=-1).mean(-1)
                        min_ade = ade.min(-1)[0].mean()
                        
                        # minFDE
                        fde = torch.norm(y_hat[..., -1, :2] - y.unsqueeze(1)[..., -1, :2], p=2, dim=-1)
                        min_fde = fde.min(-1)[0].mean()
                        
                        # MR
                        miss_threshold = 2.0
                        missed_pred = fde > miss_threshold
                        mr = missed_pred.all(-1).float().mean()
                        
                        val_loss += loss.item()
                        val_reg_loss += agent_reg_loss.item()
                        val_cls_loss += agent_cls_loss.item()
                        val_min_ade += min_ade.item()
                        val_min_fde += min_fde.item()
                        val_mr += mr.item()
                
                val_loss_avg = val_loss / len(emp_val_loader)
                val_reg_loss_avg = val_reg_loss / len(emp_val_loader)
                val_cls_loss_avg = val_cls_loss / len(emp_val_loader)
                val_min_ade_avg = val_min_ade / len(emp_val_loader)
                val_min_fde_avg = val_min_fde / len(emp_val_loader)
                val_mr_avg = val_mr / len(emp_val_loader)
                train_loss_avg = total_loss / len(emp_train_loader)
                train_reg_loss_avg = total_reg_loss / len(emp_train_loader)
                train_cls_loss_avg = total_cls_loss / len(emp_train_loader)
                
                print(f"Epoch {epoch+1}, Train Loss: {train_loss_avg:.4f} (Reg: {train_reg_loss_avg:.4f}, Cls: {train_cls_loss_avg:.4f})")
                print(f"Epoch {epoch+1}, Val Loss: {val_loss_avg:.4f} (Reg: {val_reg_loss_avg:.4f}, Cls: {val_cls_loss_avg:.4f})")
                print(f"Epoch {epoch+1}, Val minADE: {val_min_ade_avg:.4f}, Val minFDE: {val_min_fde_avg:.4f}, Val MR: {val_mr_avg:.4f}")
                
                # 记录损失和评价指标
                if 'stage3' not in loss_history:
                    loss_history['stage3'] = {'train_loss': [], 'val_loss': []}
                loss_history['stage3']['train_loss'].append(train_loss_avg)
                loss_history['stage3']['val_loss'].append(val_loss_avg)
                # 记录评价指标
                if 'val_minADE' not in loss_history['stage3']:
                    loss_history['stage3']['val_minADE'] = []
                    loss_history['stage3']['val_minFDE'] = []
                    loss_history['stage3']['val_MR'] = []
                loss_history['stage3']['val_minADE'].append(val_min_ade_avg)
                loss_history['stage3']['val_minFDE'].append(val_min_fde_avg)
                loss_history['stage3']['val_MR'].append(val_mr_avg)
                
                # 写入日志
                log_path = os.path.join(self.config['other_config']['log_dir'], f'stage3_training.log')
                with open(log_path, 'a') as f:
                    f.write(f"Epoch {epoch+1}, Train Loss: {train_loss_avg:.4f} (Reg: {train_reg_loss_avg:.4f}, Cls: {train_cls_loss_avg:.4f})\n")
                    f.write(f"Epoch {epoch+1}, Val Loss: {val_loss_avg:.4f} (Reg: {val_reg_loss_avg:.4f}, Cls: {val_cls_loss_avg:.4f})\n")
                    f.write(f"Epoch {epoch+1}, Val minADE: {val_min_ade_avg:.4f}, Val minFDE: {val_min_fde_avg:.4f}, Val MR: {val_mr_avg:.4f}\n")
                    f.write("\n")
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 保存最新模型
            save_model_fn(f"stage3_latest.pth")
            
            # 保存最佳模型（基于验证损失）
            if epoch == 0 or val_loss_avg < best_val_loss.get('stage3', float('inf')):
                best_val_loss['stage3'] = val_loss_avg
                save_model_fn(f"stage3_best.pth")
                print(f"最佳模型保存到 checkpoints/stage3_best.pth")
        
        # 训练完成后可视化损失
        visualize_fn('stage3')
        save_model_fn("stage1_best_combined.pth")
        print("联合最佳模型保存成功: stage1_best_combined.pth")



class Stage2Trainer:
    """
    Stage2: FiLM调制训练
    """
    def __init__(self, model, device, loss_fn, optimizer, config):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.scheduler = None
    
    def setup_scheduler(self):
        """
        设置学习率调度器
        """
        total_epochs = self.config['train_config']['epochs']['stage2']
        
        # 余弦退火调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs,
            eta_min=self.config['train_config']['learning_rate'] * 0.01  # 最小学习率
        )
    
    def setup_modules(self):
        """
        设置模块的冻结状态
        """
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
    
    def train(self, loss_history, best_val_loss, save_model_fn, visualize_fn):
        """
        训练FiLM模块和轨迹编码器/解码器
        """
        print("开始 Stage2: FiLM调制训练")
        
        # 设置模块
        self.setup_modules()
        
        # 设置调度器
        self.setup_scheduler()
        
        # 加载emp数据集
        emp_dataset_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/emp"
        
        # 创建数据加载器
        emp_train_loader = get_trajectory_loader(
            emp_dataset_dir, "train", 
            self.config['train_config']['batch_size']
        )
        
        emp_val_loader = get_trajectory_loader(
            emp_dataset_dir, "val", 
            self.config['train_config']['batch_size'],
            num_workers=4
        )
        
        test_emp_loader = get_trajectory_loader(
            emp_dataset_dir, "test", 
            self.config['train_config']['batch_size']
        )
        
        print(f"Stage2 - emp训练集: {len(emp_train_loader.dataset)}, 验证集: {len(emp_val_loader.dataset)}, 测试集: {len(test_emp_loader.dataset)}")
        
        # 训练FiLM模块和轨迹编码器/解码器
        for epoch in range(self.config['train_config']['epochs']['stage2']):
            self.model.train()
            total_loss = 0
            total_reg_loss = 0
            total_cls_loss = 0
            
            for i, data in enumerate(tqdm(emp_train_loader, desc=f"Stage2 Epoch {epoch+1}/{self.config['train_config']['epochs']['stage2']}")):
                # 将数据移动到设备
                data = move_trajectory_data_to_device(data, self.device)
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    # 前向传播 - 使用轨迹特征作为情绪和分心编码器的输入
                    pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(
                        trajectory_data=data, use_trajectory_feature=True
                    )
                    
                    # 计算损失（参考trainer_forecast.py的实现）
                    y_hat, pi = pred_dict['y_hat'], pred_dict['pi']
                    y = data["y"][:, 0]  # 目标车辆的真实轨迹
                    
                    B = y_hat.shape[0]
                    B_range = range(B)
                    
                    # 计算L2范数
                    l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)
                    
                    # 找到最佳预测模式
                    best_mode = torch.argmin(l2_norm, dim=-1)
                    y_hat_best = y_hat[B_range, best_mode]
                    
                    # 计算回归损失
                    agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
                    
                    # 计算分类损失
                    agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
                    
                    # 总损失
                    loss = agent_reg_loss + agent_cls_loss
                
                # 混合精度反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                total_reg_loss += agent_reg_loss.item()
                total_cls_loss += agent_cls_loss.item()
            
            # 验证
            if (epoch + 1) % self.config['other_config']['eval_interval'] == 0:
                self.model.eval()
                val_loss = 0
                val_reg_loss = 0
                val_cls_loss = 0
                val_min_ade = 0
                val_min_fde = 0
                val_mr = 0
                with torch.no_grad():
                    for data in tqdm(emp_val_loader, desc="Evaluating Stage2", leave=False):
                        # 将数据移动到设备
                        data = move_trajectory_data_to_device(data, self.device)
                        
                        # 前向传播 - 使用轨迹特征作为情绪和分心编码器的输入
                        pred_dict, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(
                            trajectory_data=data, use_trajectory_feature=True
                        )
                        
                        # 计算损失（参考trainer_forecast.py的实现）
                        y_hat, pi = pred_dict['y_hat'], pred_dict['pi']
                        y = data["y"][:, 0]  # 目标车辆的真实轨迹
                        
                        B = y_hat.shape[0]
                        B_range = range(B)
                        
                        # 计算L2范数
                        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)
                        
                        # 找到最佳预测模式
                        best_mode = torch.argmin(l2_norm, dim=-1)
                        y_hat_best = y_hat[B_range, best_mode]
                        
                        # 计算回归损失
                        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
                        
                        # 计算分类损失
                        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
                        
                        # 总损失
                        loss = agent_reg_loss + agent_cls_loss
                        
                        # 计算评价指标
                        # minADE
                        ade = torch.norm(y_hat[..., :2] - y.unsqueeze(1), p=2, dim=-1).mean(-1)
                        min_ade = ade.min(-1)[0].mean()
                        
                        # minFDE
                        fde = torch.norm(y_hat[..., -1, :2] - y.unsqueeze(1)[..., -1, :2], p=2, dim=-1)
                        min_fde = fde.min(-1)[0].mean()
                        
                        # MR
                        miss_threshold = 2.0
                        missed_pred = fde > miss_threshold
                        mr = missed_pred.all(-1).float().mean()
                        
                        val_loss += loss.item()
                        val_reg_loss += agent_reg_loss.item()
                        val_cls_loss += agent_cls_loss.item()
                        val_min_ade += min_ade.item()
                        val_min_fde += min_fde.item()
                        val_mr += mr.item()
                
                val_loss_avg = val_loss / len(emp_val_loader)
                val_reg_loss_avg = val_reg_loss / len(emp_val_loader)
                val_cls_loss_avg = val_cls_loss / len(emp_val_loader)
                val_min_ade_avg = val_min_ade / len(emp_val_loader)
                val_min_fde_avg = val_min_fde / len(emp_val_loader)
                val_mr_avg = val_mr / len(emp_val_loader)
                train_loss_avg = total_loss / len(emp_train_loader)
                train_reg_loss_avg = total_reg_loss / len(emp_train_loader)
                train_cls_loss_avg = total_cls_loss / len(emp_train_loader)
                
                print(f"Epoch {epoch+1}, Train Loss: {train_loss_avg:.4f} (Reg: {train_reg_loss_avg:.4f}, Cls: {train_cls_loss_avg:.4f})")
                print(f"Epoch {epoch+1}, Val Loss: {val_loss_avg:.4f} (Reg: {val_reg_loss_avg:.4f}, Cls: {val_cls_loss_avg:.4f})")
                print(f"Epoch {epoch+1}, Val minADE: {val_min_ade_avg:.4f}, Val minFDE: {val_min_fde_avg:.4f}, Val MR: {val_mr_avg:.4f}")
                
                # 记录损失和评价指标
                loss_history['stage2']['train_loss'].append(train_loss_avg)
                loss_history['stage2']['val_loss'].append(val_loss_avg)
                # 记录评价指标
                if 'val_minADE' not in loss_history['stage2']:
                    loss_history['stage2']['val_minADE'] = []
                    loss_history['stage2']['val_minFDE'] = []
                    loss_history['stage2']['val_MR'] = []
                loss_history['stage2']['val_minADE'].append(val_min_ade_avg)
                loss_history['stage2']['val_minFDE'].append(val_min_fde_avg)
                loss_history['stage2']['val_MR'].append(val_mr_avg)
                
                # 写入日志
                log_path = os.path.join(self.config['other_config']['log_dir'], f'stage2_training.log')
                with open(log_path, 'a') as f:
                    f.write(f"Epoch {epoch+1}, Train Loss: {train_loss_avg:.4f} (Reg: {train_reg_loss_avg:.4f}, Cls: {train_cls_loss_avg:.4f})\n")
                    f.write(f"Epoch {epoch+1}, Val Loss: {val_loss_avg:.4f} (Reg: {val_reg_loss_avg:.4f}, Cls: {val_cls_loss_avg:.4f})\n")
                    f.write(f"Epoch {epoch+1}, Val minADE: {val_min_ade_avg:.4f}, Val minFDE: {val_min_fde_avg:.4f}, Val MR: {val_mr_avg:.4f}\n")
                    f.write("\n")
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 保存最新模型
            save_model_fn(f"stage2_latest.pth")
            
            # 保存最佳模型（基于验证损失）
            if epoch == 0 or val_loss_avg < best_val_loss['stage2']:
                best_val_loss['stage2'] = val_loss_avg
                save_model_fn(f"stage2_best.pth")
                print(f"最佳模型保存到 checkpoints/stage2_best.pth")
        
        # 训练完成后可视化损失
        visualize_fn('stage2')
