import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def visualize_stage0(loss_history, log_dir):
    """
    可视化Stage0的损失曲线和评价指标
    
    Args:
        loss_history: 损失历史数据
        log_dir: 日志目录
    """
    # 创建logs目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取损失数据
    train_loss = loss_history['stage0']['train_loss']
    val_loss = loss_history['stage0']['val_loss']
    
    # 创建画布和子图
    plt.figure(figsize=(15, 6))
    
    # 左侧子图：训练和验证损失
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Stage0 Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # 右侧子图：验证评价指标
    plt.subplot(1, 2, 2)
    
    # 检查是否存在评价指标数据
    if 'val_minADE' in loss_history['stage0']:
        val_minADE = loss_history['stage0']['val_minADE']
        val_minFDE = loss_history['stage0']['val_minFDE']
        val_MR = loss_history['stage0']['val_MR']
        
        plt.plot(val_minADE, label='Val minADE')
        plt.plot(val_minFDE, label='Val minFDE')
        plt.plot(val_MR, label='Val MR')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.title('Stage0 Validation Metrics')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No validation metrics data', ha='center', va='center')
        plt.axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(log_dir, 'stage0_loss_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Stage0 损失曲线保存到 {save_path}")


def plot_confusion_matrix(ax, y_true, y_pred, title):
    """
    绘制混淆矩阵
    
    Args:
        ax: 子图对象
        y_true: 真实标签
        y_pred: 预测标签
        title: 子图标题
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制混淆矩阵
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    
    # 设置刻度
    classes = np.unique(np.concatenate([y_true, y_pred]))
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # 添加数值标签
    fmt = '.2f' if cm.max() < 1 else '.0f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")


def visualize_stage1(loss_history, log_dir):
    """
    可视化Stage1的损失曲线和混淆矩阵
    左图展示情绪部分，右图展示行为部分
    
    Args:
        loss_history: 损失历史数据
        log_dir: 日志目录
    """
    # 创建logs目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建画布和子图
    plt.figure(figsize=(15, 6))
    
    # 左侧子图：情绪部分损失
    plt.subplot(1, 2, 1)
    if 'train_loss_e' in loss_history['stage1'] and 'val_loss_e' in loss_history['stage1']:
        train_loss_e = loss_history['stage1']['train_loss_e']
        val_loss_e = loss_history['stage1']['val_loss_e']
        
        plt.plot(train_loss_e, label='Emotion Train Loss')
        plt.plot(val_loss_e, label='Emotion Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Stage1 Emotion Loss Curve')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No emotion loss data', ha='center', va='center')
        plt.axis('off')
    
    # 右侧子图：分心部分损失
    plt.subplot(1, 2, 2)
    if 'train_loss_d' in loss_history['stage1'] and 'val_loss_d' in loss_history['stage1']:
        train_loss_d = loss_history['stage1']['train_loss_d']
        val_loss_d = loss_history['stage1']['val_loss_d']
        
        plt.plot(train_loss_d, label='Distraction Train Loss')
        plt.plot(val_loss_d, label='Distraction Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Stage1 Distraction Loss Curve')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No distraction loss data', ha='center', va='center')
        plt.axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(log_dir, 'stage1_loss_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Stage1 损失曲线保存到 {save_path}")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(18, 8))
    
    # 左侧子图：情绪分类混淆矩阵
    plt.subplot(1, 2, 1)
    if 'emotion_true_labels' in loss_history['stage1'] and 'emotion_pred_labels' in loss_history['stage1']:
        emotion_true = loss_history['stage1']['emotion_true_labels']
        emotion_pred = loss_history['stage1']['emotion_pred_labels']
        if len(emotion_true) > 0 and len(emotion_pred) > 0:
            plot_confusion_matrix(plt.gca(), emotion_true, emotion_pred, 'Emotion Classification Confusion Matrix')
        else:
            plt.text(0.5, 0.5, 'No emotion confusion matrix data', ha='center', va='center')
            plt.axis('off')
    else:
        plt.text(0.5, 0.5, 'No emotion confusion matrix data', ha='center', va='center')
        plt.axis('off')
    
    # 右侧子图：分心分类混淆矩阵
    plt.subplot(1, 2, 2)
    if 'distraction_true_labels' in loss_history['stage1'] and 'distraction_pred_labels' in loss_history['stage1']:
        distraction_true = loss_history['stage1']['distraction_true_labels']
        distraction_pred = loss_history['stage1']['distraction_pred_labels']
        if len(distraction_true) > 0 and len(distraction_pred) > 0:
            plot_confusion_matrix(plt.gca(), distraction_true, distraction_pred, 'Distraction Classification Confusion Matrix')
        else:
            plt.text(0.5, 0.5, 'No distraction confusion matrix data', ha='center', va='center')
            plt.axis('off')
    else:
        plt.text(0.5, 0.5, 'No distraction confusion matrix data', ha='center', va='center')
        plt.axis('off')
    
    # 调整布局，增加底部边距
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # 保存图像
    save_path = os.path.join(log_dir, 'stage1_confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Stage1 混淆矩阵保存到 {save_path}")


def visualize_stage2(loss_history, log_dir):
    """
    可视化Stage2的损失曲线和评价指标
    留出代码块，不需要详细设计
    
    Args:
        loss_history: 损失历史数据
        log_dir: 日志目录
    """
    # 创建logs目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建画布和子图
    plt.figure(figsize=(15, 6))
    
    # 左侧子图：训练和验证损失
    plt.subplot(1, 2, 1)
    if 'train_loss' in loss_history.get('stage2', {}) and 'val_loss' in loss_history.get('stage2', {}):
        train_loss = loss_history['stage2']['train_loss']
        val_loss = loss_history['stage2']['val_loss']
        
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Stage2 Loss Curve')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No loss data', ha='center', va='center')
        plt.axis('off')
    
    # 右侧子图：验证评价指标
    plt.subplot(1, 2, 2)
    if 'val_minADE' in loss_history.get('stage2', {}):
        val_minADE = loss_history['stage2']['val_minADE']
        val_minFDE = loss_history['stage2']['val_minFDE']
        val_MR = loss_history['stage2']['val_MR']
        
        plt.plot(val_minADE, label='Val minADE')
        plt.plot(val_minFDE, label='Val minFDE')
        plt.plot(val_MR, label='Val MR')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.title('Stage2 Validation Metrics')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No validation metrics data', ha='center', va='center')
        plt.axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(log_dir, 'stage2_loss_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Stage2 损失曲线保存到 {save_path}")


def visualize_stage3(loss_history, log_dir):
    """
    可视化Stage3的损失曲线和评价指标
    与stage2相同，包含训练和验证损失、minADE、minFDE、MR
    
    Args:
        loss_history: 损失历史数据
        log_dir: 日志目录
    """
    # 创建logs目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建画布和子图
    plt.figure(figsize=(15, 6))
    
    # 左侧子图：训练和验证损失
    plt.subplot(1, 2, 1)
    if 'train_loss' in loss_history.get('stage3', {}) and 'val_loss' in loss_history.get('stage3', {}):
        train_loss = loss_history['stage3']['train_loss']
        val_loss = loss_history['stage3']['val_loss']
        
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Stage3 Loss Curve')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No loss data', ha='center', va='center')
        plt.axis('off')
    
    # 右侧子图：验证评价指标
    plt.subplot(1, 2, 2)
    if 'val_minADE' in loss_history.get('stage3', {}):
        val_minADE = loss_history['stage3']['val_minADE']
        val_minFDE = loss_history['stage3']['val_minFDE']
        val_MR = loss_history['stage3']['val_MR']
        
        plt.plot(val_minADE, label='Val minADE')
        plt.plot(val_minFDE, label='Val minFDE')
        plt.plot(val_MR, label='Val MR')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.title('Stage3 Validation Metrics')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No validation metrics data', ha='center', va='center')
        plt.axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(log_dir, 'stage3_loss_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Stage3 损失曲线保存到 {save_path}")


def visualize_loss(stage, loss_history, log_dir):
    """
    根据不同阶段调用对应的可视化函数
    
    Args:
        stage: 训练阶段名称
        loss_history: 损失历史数据
        log_dir: 日志目录
    """
    if stage == 'stage0':
        visualize_stage0(loss_history, log_dir)
    elif stage == 'stage1':
        visualize_stage1(loss_history, log_dir)
    elif stage == 'stage2':
        visualize_stage2(loss_history, log_dir)
    elif stage == 'stage3':
        visualize_stage3(loss_history, log_dir)
    else:
        print(f"不支持的训练阶段: {stage}")
