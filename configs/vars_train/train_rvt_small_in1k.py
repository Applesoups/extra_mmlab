_base_ = [
    '../_base_/models/rvt/rvt_small_in1k.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/optimizer.py', '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['vars.vars_d'
             #'vars.vision_transformer_vars'
            ], allow_failed_imports=False)

checkpoint_config = dict(interval=1)
