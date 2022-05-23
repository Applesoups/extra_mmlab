#model settings for nat-tiny
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='NAT',
        embed_dim=96,
        mlp_ratio=2,
        depths=[3, 4, 18, 5],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.3,
        in_chans=3,
        kernel_size=7,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        layer_scale=1e-5,
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=96*2**(4-1),
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1),
    ),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, prob=0.5, num_classes=1000),
        dict(type='BatchCutMix', alpha=1.0, prob=0.5, num_classes=1000),
    ])
)
