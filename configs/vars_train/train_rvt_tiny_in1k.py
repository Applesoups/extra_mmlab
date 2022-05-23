_base_ = [
    '../_base_/models/rvt/rvt_tiny_vars_d_in1k.py',
    '../_base_mmcls/datasets/imagenet_bs64_swin_224.py',
    '../_base_mmcls/schedules/imagenet_bs1024_adamw_swin.py', '../_base_mmcls/default_runtime.py'
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        # data_prefix='/home/intern/dataset/imagenet/train',
        ann_file=None),
    val=dict(
        # data_prefix='/home/intern/dataset/imagenet/val',
        ann_file=None),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        # data_prefix='/home/intern/dataset/imagenet/val',
        ann_file=None)
)



optimizer = dict(
    type='AdamW',
    lr=5e-4 * 256 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999))
#optimizer_config = dict(grad_clip=dict(max_norm=5.0))
optimizer_config = dict()

lr_config = dict(
    policy='CosineAnnealing',
    #policy='CosineRestart',
    by_epoch=False,
    min_lr=1e-5,
    min_lr_ratio=None,
    warmup='linear',
    warmup_ratio=2e-3,
     # LR used at the beginning of warmup equals to warmup_ratio * initial_lr , as vars warmup-lr is 1e-6, here i set warmup_ratio as 2e-3
    warmup_iters=5,
    warmup_by_epoch=True)



custom_imports = dict(
    imports=['vars.vars_d',
             #'vars.vision_transformer_vars',
             'vars.dataset'
             ],
    allow_failed_imports=False)



checkpoint_config = dict(interval=1)
