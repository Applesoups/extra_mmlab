_base_ = [
    '../_base_/models/nat/nat_small.py',
    '../_base_mmcls/datasets/imagenet_bs64_swin_224.py',
    '../_base_mmcls/schedules/imagenet_bs1024_adamw_swin.py', '../_base_mmcls/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

batch_size=32

data = dict(
    samples_per_gpu=batch_size,
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
    #lr=5e-4 * batch_size / 512,
    lr=1e-3 * batch_size / 512,
    weight_decay=0.1,
    eps=1e-8,
    betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

lr_config = dict(
    policy='CosineAnnealing',
    #policy='CosineRestart',
    by_epoch=False,
    min_lr=5e-6,
    min_lr_ratio=None,
    warmup='linear',
    warmup_ratio=1e-3,
     # LR used at the beginning of warmup equals to warmup_ratio * initial_lr , as vars warmup-lr is 1e-6, here i set warmup_ratio as 1e-3
    warmup_iters=20,
    warmup_by_epoch=True)

custom_imports = dict(
    imports=['NAT.nat'],
    allow_failed_imports=False
)