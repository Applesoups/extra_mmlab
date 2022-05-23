model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='cait',
        img_size= 224,
        embed_dim=192, 
        depth=24, 
        num_heads=4, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer_type='partial',
        init_scale=1e-5,
        depth_token_only=2),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=192,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1),
    ),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, prob=0.5, num_classes=1000),
        dict(type='BatchCutMix', alpha=1.0, prob=0.5, num_classes=1000),
    ])
)