#model settings for nat-tiny
depth = 12
sr_ratio = [1] * depth
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='FAN',
        patch_size=16, 
        embed_dim=384, 
        depth=depth, 
        num_heads=8, 
        eta=1.0, 
        tokens_norm=True, 
        sr_ratio=sr_ratio
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=384,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1),
    ),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, prob=0.5, num_classes=1000),
        dict(type='BatchCutMix', alpha=1.0, prob=0.5, num_classes=1000),
    ])
)
