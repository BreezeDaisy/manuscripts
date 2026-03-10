import torch
import torch.nn as nn
from .trajectory_loss import TrajectoryLoss
from .vae_loss import VAELoss
from .orthogonal_loss import OrthogonalLoss


class TotalLoss(nn.Module):
    """总损失函数"""
    def __init__(self, lambda_traj=1.0, lambda_kl_e=1.0, lambda_kl_d=1.0, lambda_orth=1.0):
        """
        初始化总损失函数
        
        Args:
            lambda_traj: 轨迹损失权重
            lambda_kl_e: 情绪KL散度损失权重
            lambda_kl_d: 分心KL散度损失权重
            lambda_orth: 正交损失权重
        """
        super(TotalLoss, self).__init__()
        
        # 初始化各个损失函数
        self.trajectory_loss = TrajectoryLoss()
        self.vae_loss = VAELoss()
        self.orthogonal_loss = OrthogonalLoss()
        
        # 权重参数
        self.lambda_traj = lambda_traj
        self.lambda_kl_e = lambda_kl_e
        self.lambda_kl_d = lambda_kl_d
        self.lambda_orth = lambda_orth
    
    def forward(self, pred, target, z_e, z_d, mu_e, logvar_e, mu_d, logvar_d):
        """
        计算总损失
        
        L = L_traj + λ1 L_emotion + λ2 L_distraction + λ3 KL_e + λ4 KL_d + λ5 L_orth
        
        Args:
            pred: 预测的轨迹 [batch_size, future_steps, output_dim]
            target: 真实的轨迹 [batch_size, future_steps, output_dim]
            z_e: 情绪潜在表示 [batch_size, latent_dim]
            z_d: 分心潜在表示 [batch_size, latent_dim]
            mu_e: 情绪VAE均值 [batch_size, latent_dim]
            logvar_e: 情绪VAE对数方差 [batch_size, latent_dim]
            mu_d: 分心VAE均值 [batch_size, latent_dim]
            logvar_d: 分心VAE对数方差 [batch_size, latent_dim]
            
        Returns:
            total_loss: 总损失
            traj_loss: 轨迹损失
            kl_loss_e: 情绪KL散度损失
            kl_loss_d: 分心KL散度损失
            orth_loss: 正交损失
        """
        # 计算轨迹损失
        traj_loss = self.trajectory_loss(pred, target)
        
        # 计算情绪VAE的KL散度损失
        kl_loss_e = self.vae_loss(mu_e, logvar_e)
        
        # 计算分心VAE的KL散度损失
        kl_loss_d = self.vae_loss(mu_d, logvar_d)
        
        # 计算正交损失
        orth_loss = self.orthogonal_loss(z_e, z_d)
        
        # 计算总损失
        total_loss = self.lambda_traj * traj_loss + \
                     self.lambda_kl_e * kl_loss_e + \
                     self.lambda_kl_d * kl_loss_d + \
                     self.lambda_orth * orth_loss
        
        return total_loss, traj_loss, kl_loss_e, kl_loss_d, orth_loss