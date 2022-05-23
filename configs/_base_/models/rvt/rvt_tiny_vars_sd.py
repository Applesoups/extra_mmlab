#model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PoolingTransformer',
        image_size=224,
        patch_size=16,
        base_dims=[32, 32],
        depth=[10, 2],
        heads=[6, 12],
        in_chans=3,
        attention_type='vars_sd',
        drop_rate=.0,  
        drop_path_rate=.0,
        proj_drop=.0,
        mlp_ratio=4,
        attn_cfgs=dict(attn_drop=.0,
                rand_feat_dim_ratio=2,
                lam=0.3,
                num_step=5,
                qkv_bias=True,
                qk_scale=None
                ),
        layer_cfgs=dict(act_layer='Gelu',
                norm_layer='Partial'),
        masked_cfgs=dict(use_mask=False,
                masked_block=None)
        ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=32*12,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #topk=(1, 5),
    )
)