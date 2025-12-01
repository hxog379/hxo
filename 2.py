# configs/simple_crack_config.py
# 简化版配置 - 基于DeepLabV3
_base_ = [
    'mmseg::deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py'
]

# 修改为二分类（背景+裂缝）
model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2)
)

# 简化数据增强
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

