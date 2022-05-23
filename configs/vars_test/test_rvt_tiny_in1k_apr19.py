_base_ = [
    '../_base_/models/rvt/rvt_tiny_vars_d_in1k.py',
    '../_base_/datasets/imagenet_bs96_test_mmlab.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py', '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['vars.vars_d',
             #'vars.vision_transformer_vars',
             'vars.dataset'
             ],
    allow_failed_imports=False)

checkpoint_config = dict(interval=1)
