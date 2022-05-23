#model settings for nat-tiny
depth = 23
stage_depth = 20
channel_dims = [528] * stage_depth + [768] * (depth - stage_depth)
num_heads = [11] * stage_depth + [16] * (depth - stage_depth)
sr_ratio = [1] * (depth//2) + [1] * (depth//2 +1)
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='FAN',
        patch_size=16, 
        embed_dim=channel_dims[0], 
        depth=depth, 
        num_heads=num_heads, 
        eta=1.0, 
        tokens_norm=True, 
        sharpen_attn=False,
        sr_ratio=sr_ratio,
        channel_dims=channel_dims,
        model_args = dict(depths=[3, 7], 
        dims=[128, 256, 512, 1024], 
        use_head=False)
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=channel_dims[depth-1],
        init_cfg=dict(type='Normal', layer='Linear', std=0.0002),
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1),
    ),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, prob=0.5, num_classes=1000),
        dict(type='BatchCutMix', alpha=1.0, prob=0.5, num_classes=1000),
    ])
)
