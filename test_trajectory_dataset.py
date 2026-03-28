import torch
from datasets.trajectory_dataset import TrajectoryDataset

# 配置参数
data_root = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/emp"
split = "train"
seq_len = 50  # 与config.py中的seq_len一致
future_steps = 60  # 与config.py中的future_steps一致
input_dim = 4

# 创建数据集实例
dataset = TrajectoryDataset(
    data_root=data_root,
    split=split,
    seq_len=seq_len,
    future_steps=future_steps,
    input_dim=input_dim
)

# 输出数据集信息
print(f"数据集长度: {len(dataset)}")
print(f"文件列表长度: {len(dataset.file_list)}")
print(f"文件列表前5个: {[str(f) for f in dataset.file_list[:5]]}")

# 测试获取数据
if len(dataset) > 0:
    print("\n测试获取数据...")
    # 获取第一个数据项
    try:
        history, future = dataset[0]
        print(f"历史轨迹形状: {history.shape}")
        print(f"未来轨迹形状: {future.shape}")
        print(f"历史轨迹前5个点:")
        print(history[:5])
        print(f"未来轨迹前5个点:")
        print(future[:5])
        
        # 检查数据是否为随机生成
        if torch.allclose(history.mean(), torch.tensor(0.0), atol=1e-1) and torch.allclose(history.std(), torch.tensor(1.0), atol=1e-1):
            print("\n警告: 数据可能是随机生成的！")
        else:
            print("\n数据看起来是真实的！")
    except Exception as e:
        print(f"获取数据时出错: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n数据集为空！")

# 测试文件加载
print("\n测试文件加载...")
if len(dataset.file_list) > 0:
    test_file = dataset.file_list[0]
    print(f"测试文件: {test_file}")
    try:
        data = torch.load(test_file)
        print(f"文件加载成功！")
        print(f"文件中的键: {list(data.keys())}")
        if "history" in data:
            print(f"history形状: {data['history'].shape}")
        if "future" in data:
            print(f"future形状: {data['future'].shape}")
    except Exception as e:
        print(f"加载文件时出错: {e}")
        import traceback
        traceback.print_exc()
else:
    print("没有文件可测试！")

# 测试批量获取
print("\n测试批量获取数据...")
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

try:
    for i, (history_batch, future_batch) in enumerate(loader):
        print(f"批次 {i+1}:")
        print(f"历史轨迹批次形状: {history_batch.shape}")
        print(f"未来轨迹批次形状: {future_batch.shape}")
        # 只打印第一个批次
        if i == 0:
            print(f"第一个样本的历史轨迹前3个点:")
            print(history_batch[0][:3])
            print(f"第一个样本的未来轨迹前3个点:")
            print(future_batch[0][:3])
        break
except Exception as e:
    print(f"批量获取数据时出错: {e}")
    import traceback
    traceback.print_exc()