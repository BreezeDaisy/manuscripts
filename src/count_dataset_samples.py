import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 相对导入只能使用 python -m 运行，添加临时搜索路径的方法兼容性最好
from emotion_dataset import EmotionDataset
from distraction_dataset import DistractionDataset
from trajectory_dataset import TrajectoryDataset

# 统计情绪数据集
def count_emotion_dataset():
    print("\n=== 统计情绪数据集 ===")
    
    # 训练集
    try:
        train_dataset = EmotionDataset(
            dataset_dir="/home/zdx/python_daima/MVim/MVim/Face_Pose/small_data",
            split="train",
            transform=None
        )
        print(f"训练集样本数: {len(train_dataset)}")
        
        # 统计类别分布
        label_counts = {}
        for label in train_dataset.labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        # 映射标签到情绪名称
        emotion_names = {
            0: "antipathic",
            1: "fear",
            2: "happy",
            3: "neutral",
            4: "sad",
            5: "surprise"
        }
        
        print("训练集类别分布:")
        for label, count in sorted(label_counts.items()):
            print(f"  {emotion_names[label]}: {count}")
    except Exception as e:
        print(f"加载训练集失败: {e}")
    
    # 验证集
    try:
        val_dataset = EmotionDataset(
            dataset_dir="/home/zdx/python_daima/MVim/MVim/Face_Pose/small_data",
            split="val",
            transform=None
        )
        print(f"\n验证集样本数: {len(val_dataset)}")
        
        # 统计类别分布
        label_counts = {}
        for label in val_dataset.labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        print("验证集类别分布:")
        for label, count in sorted(label_counts.items()):
            print(f"  {emotion_names[label]}: {count}")
    except Exception as e:
        print(f"加载验证集失败: {e}")

# 统计分心数据集
def count_distraction_dataset():
    print("\n=== 统计分心数据集 ===")
    
    try:
        # 假设数据集路径
        root_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/SFDDD"
        
        # 训练集
        train_dataset = DistractionDataset(
            root_dir=root_dir,
            split="train",
            transform=None
        )
        print(f"训练集样本数: {len(train_dataset)}")
        
        # 统计类别分布
        label_counts = {}
        for label in train_dataset.labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        # 映射标签到类别名称
        class_names = list(train_dataset.class_names.values())
        
        print("训练集类别分布:")
        for label, count in sorted(label_counts.items()):
            if 0 <= label < len(class_names):
                print(f"  {class_names[label]}: {count}")
        
        # 验证集
        val_dataset = DistractionDataset(
            root_dir=root_dir,
            split="val",
            transform=None
        )
        print(f"\n验证集样本数: {len(val_dataset)}")
        
        # 统计类别分布
        label_counts = {}
        for label in val_dataset.labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        print("验证集类别分布:")
        for label, count in sorted(label_counts.items()):
            if 0 <= label < len(class_names):
                print(f"  {class_names[label]}: {count}")
    except Exception as e:
        print(f"加载分心数据集失败: {e}")


if __name__ == "__main__":
    count_emotion_dataset()
    count_distraction_dataset()
 
