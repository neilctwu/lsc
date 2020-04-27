import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url

__all__ = ['EfficientNetB0']
model_url = ''

#TODO: forward, build

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000, init_weight=True, pretrain=False):
        super(EfficientNetB0, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        self._build(num_classes)
        # automatically abandon init_weight if pretrain is True
        if pretrain:
            assert model_url is not '', f'Pretrained model for {self.__class__.__name__} not prepared yet.'
            state_dict = load_state_dict_from_url(model_url,
                                                  progress=True)
            self.load_state_dict(state_dict)
        elif init_weight:
            self._init_weights()

    def _build(self, num_classes):
        pass

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        pass

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

