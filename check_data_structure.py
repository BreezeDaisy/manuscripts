import torch
import os

# 数据文件路径
data_dir = "/home/zdx/python_daima/MVim/manuscripts/datasets/dataset/emp/train"

# 获取数据文件列表
file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
if not file_list:
    print("没有找到数据文件")
    exit(1)

# 选择一个数据文件进行检查
test_file = os.path.join(data_dir, file_list[0])
print(f"检查文件: {test_file}")

# 加载数据文件
try:
    data = torch.load(test_file)
    print("文件加载成功！")
    print(f"数据类型: {type(data)}")
    
    if isinstance(data, dict):
        print("\n数据文件包含的键:")
        for key in data.keys():
            value = data[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
            else:
                print(f"  {key}: 类型={type(value)}")
    else:
        print("数据文件不是字典格式")
        print(f"数据内容: {data}")
except Exception as e:
    print(f"加载文件失败: {e}")

# 检查多个文件以确保一致性
print("\n检查多个文件以确保一致性...")
for i, file_name in enumerate(file_list[:5]):  # 检查前5个文件
    file_path = os.path.join(data_dir, file_name)
    try:
        data = torch.load(file_path)
        if isinstance(data, dict):
            keys = sorted(data.keys())
            print(f"文件 {i+1} 键数量: {len(keys)}")
            print(f"  键: {keys}")
        else:
            print(f"文件 {i+1} 不是字典格式")
    except Exception as e:
        print(f"文件 {i+1} 加载失败: {e}")
