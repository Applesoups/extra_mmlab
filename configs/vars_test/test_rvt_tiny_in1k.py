_base_ = [
    '../_base_/models/rvt/rvt_tiny_vars_d.py',
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/optimizer.py', '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['vars.robust_models',
             #'vars.vision_transformer_vars',
             'vars.dataset'
             ],
    allow_failed_imports=False)

checkpoint_config = dict(interval=1)
