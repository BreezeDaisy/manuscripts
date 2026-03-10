import torch
import torch.optim as optim
import os
from tqdm import tqdm
from ..losses.total_loss import TotalLoss
from ..datasets.dataset import get_dataloader

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
        self.device = config['train_config']['device']
        self.model.to(self.device)
        
        # 损失函数
        self.loss_fn = TotalLoss(
            lambda_traj=config['loss_config']['lambda_traj'],
            lambda_kl_e=config['loss_config']['lambda_kl_e'],
            lambda_kl_d=config['loss_config']['lambda_kl_d'],
            lambda_orth=config['loss_config']['lambda_orth']
        )
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['train_config']['learning_rate'],
            weight_decay=config['train_config']['weight_decay']
        )
        
        # 数据加载器
        self.train_loader = get_dataloader(
            num_samples=config['data_config']['num_samples'],
            batch_size=config['train_config']['batch_size'],
            shuffle=config['data_config']['shuffle'],
            seq_len=config['model_config']['seq_len'],
            future_steps=config['model_config']['future_steps'],
            input_dim=config['model_config']['trajectory_input_dim'],
            emotion_num_classes=config['model_config']['emotion_num_classes'],
            distraction_num_classes=config['model_config']['distraction_num_classes']
        )
        
        self.val_loader = get_dataloader(
            num_samples=int(config['data_config']['num_samples'] * config['data_config']['val_split']),
            batch_size=config['train_config']['batch_size'],
            shuffle=False,
            seq_len=config['model_config']['seq_len'],
            future_steps=config['model_config']['future_steps'],
            input_dim=config['model_config']['trajectory_input_dim'],
            emotion_num_classes=config['model_config']['emotion_num_classes'],
            distraction_num_classes=config['model_config']['distraction_num_classes']
        )
        
        # 创建检查点目录
        os.makedirs(config['other_config']['checkpoint_dir'], exist_ok=True)
    
    def train_stage0(self):
        """
        Stage0: 轨迹网络预训练
        """
        print("开始 Stage0: 轨迹网络预训练")
        
        # 冻结情绪和分心编码器
        for param in self.model.emotion_encoder.parameters():
            param.requires_grad = False
        
        for param in self.model.distraction_encoder.parameters():
            param.requires_grad = False
        
        # 冻结FiLM模块
        for param in self.model.film_module.parameters():
            param.requires_grad = False
        
        # 训练轨迹编码器和解码器
        for epoch in range(self.config['train_config']['epochs']['stage0']):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(self.train_loader, desc=f"Stage0 Epoch {epoch+1}/{self.config['train_config']['epochs']['stage0']}"):
                x_e, label_e, x_d, label_d, x_t, y = batch
                x_e, label_e, x_d, label_d, x_t, y = x_e.to(self.device), label_e.to(self.device), x_d.to(self.device), label_d.to(self.device), x_t.to(self.device), y.to(self.device)
                
                # 前向传播
                pred, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(x_e, label_e, x_d, label_d, x_t)
                
                # 计算损失（只计算轨迹损失）
                loss = self.loss_fn.trajectory_loss(pred, y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # 验证
            if (epoch + 1) % self.config['other_config']['eval_interval'] == 0:
                val_loss = self.evaluate()
                print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存模型
            if (epoch + 1) % self.config['other_config']['save_interval'] == 0:
                self.save_model(f"stage0_epoch{epoch+1}.pth")
    
    def train_stage1(self):
        """
        Stage1: 情绪和分心编码器训练
        """
        print("开始 Stage1: 情绪和分心编码器训练")
        
        # 解冻情绪和分心编码器
        for param in self.model.emotion_encoder.parameters():
            param.requires_grad = True
        
        for param in self.model.distraction_encoder.parameters():
            param.requires_grad = True
        
        # 冻结轨迹编码器和解码器
        for param in self.model.trajectory_encoder.parameters():
            param.requires_grad = False
        
        for param in self.model.trajectory_decoder.parameters():
            param.requires_grad = False
        
        # 冻结FiLM模块
        for param in self.model.film_module.parameters():
            param.requires_grad = False
        
        # 训练情绪和分心编码器
        for epoch in range(self.config['train_config']['epochs']['stage1']):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(self.train_loader, desc=f"Stage1 Epoch {epoch+1}/{self.config['train_config']['epochs']['stage1']}"):
                x_e, label_e, x_d, label_d, x_t, y = batch
                x_e, label_e, x_d, label_d, x_t, y = x_e.to(self.device), label_e.to(self.device), x_d.to(self.device), label_d.to(self.device), x_t.to(self.device), y.to(self.device)
                
                # 前向传播
                pred, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(x_e, label_e, x_d, label_d, x_t)
                
                # 计算损失（主要是VAE损失）
                _, traj_loss, kl_loss_e, kl_loss_d, _ = self.loss_fn(pred, y, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d)
                loss = kl_loss_e + kl_loss_d
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # 验证
            if (epoch + 1) % self.config['other_config']['eval_interval'] == 0:
                val_loss = self.evaluate()
                print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存模型
            if (epoch + 1) % self.config['other_config']['save_interval'] == 0:
                self.save_model(f"stage1_epoch{epoch+1}.pth")
    
    def train_stage2(self):
        """
        Stage2: FiLM调制训练
        """
        print("开始 Stage2: FiLM调制训练")
        
        # 解冻FiLM模块
        for param in self.model.film_module.parameters():
            param.requires_grad = True
        
        # 保持其他模块冻结状态
        for param in self.model.emotion_encoder.parameters():
            param.requires_grad = False
        
        for param in self.model.distraction_encoder.parameters():
            param.requires_grad = False
        
        for param in self.model.trajectory_encoder.parameters():
            param.requires_grad = False
        
        for param in self.model.trajectory_decoder.parameters():
            param.requires_grad = False
        
        # 训练FiLM模块
        for epoch in range(self.config['train_config']['epochs']['stage2']):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(self.train_loader, desc=f"Stage2 Epoch {epoch+1}/{self.config['train_config']['epochs']['stage2']}"):
                x_e, label_e, x_d, label_d, x_t, y = batch
                x_e, label_e, x_d, label_d, x_t, y = x_e.to(self.device), label_e.to(self.device), x_d.to(self.device), label_d.to(self.device), x_t.to(self.device), y.to(self.device)
                
                # 前向传播
                pred, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(x_e, label_e, x_d, label_d, x_t)
                
                # 计算损失（主要是轨迹损失）
                total_loss_batch, traj_loss, kl_loss_e, kl_loss_d, _ = self.loss_fn(pred, y, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d)
                loss = traj_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # 验证
            if (epoch + 1) % self.config['other_config']['eval_interval'] == 0:
                val_loss = self.evaluate()
                print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存模型
            if (epoch + 1) % self.config['other_config']['save_interval'] == 0:
                self.save_model(f"stage2_epoch{epoch+1}.pth")
    
    def train_stage3(self):
        """
        Stage3: 联合微调
        """
        print("开始 Stage3: 联合微调")
        
        # 解冻所有模块
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 联合微调所有模块
        for epoch in range(self.config['train_config']['epochs']['stage3']):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(self.train_loader, desc=f"Stage3 Epoch {epoch+1}/{self.config['train_config']['epochs']['stage3']}"):
                x_e, label_e, x_d, label_d, x_t, y = batch
                x_e, label_e, x_d, label_d, x_t, y = x_e.to(self.device), label_e.to(self.device), x_d.to(self.device), label_d.to(self.device), x_t.to(self.device), y.to(self.device)
                
                # 前向传播
                pred, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(x_e, label_e, x_d, label_d, x_t)
                
                # 计算总损失
                total_loss_batch, traj_loss, kl_loss_e, kl_loss_d, _ = self.loss_fn(pred, y, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d)
                loss = total_loss_batch
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # 验证
            if (epoch + 1) % self.config['other_config']['eval_interval'] == 0:
                val_loss = self.evaluate()
                print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存模型
            if (epoch + 1) % self.config['other_config']['save_interval'] == 0:
                self.save_model(f"stage3_epoch{epoch+1}.pth")
    
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
                x_e, label_e, x_d, label_d, x_t, y = batch
                x_e, label_e, x_d, label_d, x_t, y = x_e.to(self.device), label_e.to(self.device), x_d.to(self.device), label_d.to(self.device), x_t.to(self.device), y.to(self.device)
                
                # 前向传播
                pred, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d = self.model(x_e, label_e, x_d, label_d, x_t)
                
                # 计算总损失
                total_loss_batch, _, _, _, _ = self.loss_fn(pred, y, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d)
                total_loss += total_loss_batch.item()
        
        return total_loss / len(self.val_loader)
    
    def save_model(self, filename):
        """
        保存模型
        
        Args:
            filename: 文件名
        """
        save_path = os.path.join(self.config['other_config']['checkpoint_dir'], filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"模型保存到 {save_path}")
    
    def load_model(self, filename):
        """
        加载模型
        
        Args:
            filename: 文件名
        """
        load_path = os.path.join(self.config['other_config']['checkpoint_dir'], filename)
        self.model.load_state_dict(torch.load(load_path))
        print(f"从 {load_path} 加载模型")

# 导入配置
from ..configs.config import model_config, train_config, data_config, loss_config, other_config
config = {
    'model_config': model_config,
    'train_config': train_config,
    'loss_config': loss_config,
    'data_config': data_config,
    'other_config': other_config
}