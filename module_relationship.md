# 驾驶行为调制模型模块调用关系图

## 模块调用关系

```
main.py
├── models/full_model.py
│   ├── models/emotion_encoder.py
│   │   └── utils/reparameterize.py
│   ├── models/distraction_encoder.py
│   │   └── utils/reparameterize.py
│   ├── models/trajectory_encoder.py
│   ├── models/trajectory_decoder.py
│   └── models/film_module.py
├── trainer/trainer.py
│   ├── models/full_model.py
│   ├── losses/total_loss.py
│   │   ├── losses/trajectory_loss.py
│   │   ├── losses/vae_loss.py
│   │   └── losses/orthogonal_loss.py
│   └── datasets/dataset.py
│       ├── datasets/emotion_dataset.py
│       ├── datasets/distraction_dataset.py
│       └── datasets/trajectory_dataset.py
├── utils/seed.py
├── utils/logger.py
└── configs/config.py

scripts/train_stage0.py
├── models/full_model.py
├── trainer/trainer.py
├── utils/seed.py
├── utils/logger.py
└── configs/config.py

scripts/train_stage1.py
├── models/full_model.py
├── trainer/trainer.py
├── utils/seed.py
├── utils/logger.py
└── configs/config.py

scripts/train_stage2.py
├── models/full_model.py
├── trainer/trainer.py
├── utils/seed.py
├── utils/logger.py
└── configs/config.py

scripts/train_stage3.py
├── models/full_model.py
├── trainer/trainer.py
├── utils/seed.py
├── utils/logger.py
└── configs/config.py

configs/config.py
├── models/full_model.py
├── trainer/trainer.py
├── scripts/train_stage0.py
├── scripts/train_stage1.py
├── scripts/train_stage2.py
├── scripts/train_stage3.py
└── main.py

utils/reparameterize.py
├── models/emotion_encoder.py
└── models/distraction_encoder.py

utils/seed.py
├── scripts/train_stage0.py
├── scripts/train_stage1.py
├── scripts/train_stage2.py
├── scripts/train_stage3.py
└── main.py

utils/logger.py
├── scripts/train_stage0.py
├── scripts/train_stage1.py
├── scripts/train_stage2.py
├── scripts/train_stage3.py
└── main.py
```

## 训练流程

### Stage0: 轨迹网络预训练
1. 冻结情绪和分心编码器
2. 冻结FiLM模块
3. 训练轨迹编码器和解码器

### Stage1: 情绪和分心编码器训练
1. 解冻情绪和分心编码器
2. 冻结轨迹编码器和解码器
3. 冻结FiLM模块
4. 训练情绪和分心编码器（主要优化VAE损失）

### Stage2: FiLM调制训练
1. 解冻FiLM模块
2. 保持其他模块冻结状态
3. 训练FiLM模块（主要优化轨迹损失）

### Stage3: 联合微调
1. 解冻所有模块
2. 联合微调所有模块（优化总损失）

## 数据流向

1. 输入数据：
   - 情绪特征 (x_e)
   - 分心特征 (x_d)
   - 历史轨迹 (x_t)

2. 处理流程：
   - 情绪编码器：x_e → z_e (VAE潜在表示)
   - 分心编码器：x_d → z_d (VAE潜在表示)
   - 轨迹编码器：x_t → h (轨迹特征)
   - FiLM模块：h, z_e, z_d → h' (调制后的轨迹特征)
   - 轨迹解码器：h' → 预测未来轨迹

3. 输出：
   - 预测的未来轨迹
