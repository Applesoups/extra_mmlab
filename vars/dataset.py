from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mmcls.datasets import PIPELINES
import cv2
from PIL import Image
import numpy as np

def build_transform(is_train, 
                    input_size,
                    color_jitter,
                    auto_augment,
                    interpolation,
                    re_prob,
                    re_mode,
                    re_count):
    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size,
            is_train,
            color_jitter,
            auto_augment,
            interpolation,
            re_prob,
            re_mode,
            re_count,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform
    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


@PIPELINES.register_module()
class mytransforms(object):
    def __init__(self,
                input_size,
                is_train,
                color_jitter,
                auto_augment,
                interpolation,
                re_prob,
                re_mode,
                re_count):
        self.transform=build_transform(is_train,input_size,color_jitter,
                                       auto_augment, interpolation, re_prob, re_mode,re_count)
    
    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img=self.transform(img)
            img=np.array(img)
            results[key]=img
        results['image_shape']=img.shape
        return results

