import os
import time
import mmcv
import torch
import inspect
import cv2
import collections
import numpy as np
from torch import nn
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result
from mmdet.models.registry import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                                   ROI_EXTRACTORS, SHARED_HEADS)


class Compose(object):

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


# def inference_detector(model, img):
#     cfg = model.cfg
#     device = next(model.parameters()).device  # model device
#     # build the data pipeline
#     test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
#     test_pipeline = Compose(test_pipeline)
#     # prepare data
#     data = dict(img=img)
#     data = test_pipeline(data)
#     data = scatter(collate([data], samples_per_gpu=1), [device])[0]
#     # forward the model
#     with torch.no_grad():
#         result = model(return_loss=False, rescale=True, **data)
#     return result


def build_from_cfg(cfg, registry, default_args=None):
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def init_detector(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)

    config.model.pretrained = None
    model = build(config.model, DETECTORS, dict(
        train_cfg=None, test_cfg=config.test_cfg))

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def detectAPI(config, modelfile, gpuid=0):
    model = init_detector(config, modelfile, device='cuda:{}'.format(gpuid))
    return model

def bgr2gray(img, keepdim=False):
    out_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if keepdim:
        out_img = out_img[..., None]
    return out_img

def convert_color_factory(src, dst):

    code = getattr(cv2, 'COLOR_{}2{}'.format(src.upper(), dst.upper()))

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = """Convert a {0} image to {1} image.
    Args:
        img (ndarray or str): The input image.
    Returns:
        ndarray: The converted {1} image.
    """.format(src.upper(), dst.upper())

    return convert_color

bgr2rgb = convert_color_factory('bgr', 'rgb')

def imnormalize(img, mean, std, to_rgb=True):
    img = img.astype(np.float32)
    if to_rgb:
        img = bgr2rgb(img)
    return (img - mean) / std



# class Model():

#     def __init__(self):
        
#         pass
    
#     def 





def main():


    mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
    std = np.array([1., 1., 1.], dtype=np.float32)
    '''
            img = cv2.imread(imgpath)
            results['img'] = img
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape

            size = (300, 300)
            h, w = img.shape[:2]
            rim = cv2.resize(img, size)
            w_scale = size[0] / w
            h_scale = size[1] / h

            results['img'] = imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb

            results['img_norm_cfg'] = dict(
            mean=mean, std, to_rgb=True)

            data = torch.from_numpy(results['img'])


            model = build(config.model, DETECTORS, dict(
                train_cfg=None, test_cfg=config.test_cfg))

            if checkpoint is not None:
                checkpoint = load_checkpoint(model, checkpoint)

    '''

    gpuid = 0
    config = '/data/code/sunchao/mmdetection/my_config/skudet/skudet_ssd300_mobilev2_2x_200103_d2.py'
    modelfile = '/data/code/sunchao/mmdetection/epoch_23.pth'
    # detAPI = detectAPI(config, modelfile, gpuid)
    detAPI = init_detector(config, modelfile, device='cuda:{}'.format(gpuid))
    class_names = ('3477')
    testDir = '/home/train/Desktop/demo/del/JPEGImages/'
    imgList = os.listdir(testDir)
    alltime = 0

    for idx, filename in enumerate(imgList):

        print(idx, filename)
        filePath = os.path.join(testDir, filename)

        filePath = '/home/train/Desktop/demo/0d75478572370596c6df01225a4b982d.jpg'
        stime = time.time()
        result = inference_detector(detAPI, filePath)
        etime = time.time()
        utime = etime - stime
        alltime += utime
        show_result(filePath, result, detAPI.CLASSES)
    # print("use time: {}/{}={}".format(alltime, len(imgList), alltime/(len(imgList)*1.)))
        exit()

if __name__ == "__main__":
    main()
