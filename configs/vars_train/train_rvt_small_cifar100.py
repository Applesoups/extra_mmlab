_base_ = [
    '../_base_/models/rvt/rvt_small_cifar100.py',
    '../_base_/datasets/cifar100_bs16.py',
    '../_base_/schedules/optimizer.py', '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['vars.vars_d'
             #'vars.vision_transformer_vars'
            ], allow_failed_imports=False)

checkpoint_config = dict(interval=1)
