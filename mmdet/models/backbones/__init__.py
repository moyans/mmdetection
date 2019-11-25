from .hrnet import HRNet
from .resnet import ResNet, make_res_layer
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .efficientnet import EfficientNet
from .simple_convnet import ConvnetLprVehicle, ConvnetLprPlate
from .senet import SENet

# __all__ = ['ResNet', 'make_res_layer', 'ResNeXt', 'SSDVGG', 'HRNet']

__all__ = ['ResNet', 'make_res_layer', 'ResNeXt', 'SSDVGG', 'HRNet',
            'EfficientNet', 'ConvnetLprVehicle', 'ConvnetLprPlate', 'SENet']


'''
copyright@https://github.com/open-mmlab/mmdetection/issues/111
You can add new backbones under models/backbones, e.g., unet.py, and then specify it in config files.

unet.py

class UNet(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


config file
model = dict(backbone=(type='UNet', xxxx), xxxx)
'''