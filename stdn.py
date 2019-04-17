# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from utils.core import print_info


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    return model


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, predict_layer=(5, 10, 15, 20, 25, 32), bn_size=4, drop_rate=0):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(num_init_features, num_init_features,
                                kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True)),

            ('conv2', nn.Conv2d(num_init_features, num_init_features,
                                kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(num_init_features)),
            ('relu2', nn.ReLU(inplace=True)),

            ('pool2', nn.AvgPool2d(kernel_size=2, stride=2)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.predict_layer = predict_layer

    def forward(self, x):
        x = self.features[0:-1](x)
        soureces = []
        denseblock4_len = len(self.features[-1])
        for i in xrange(denseblock4_len):
            x = self.features[-1][i](x)
            if (i + 1) in self.predict_layer:
                soureces += [x]
        return soureces


class Scale_Transfer_Layer(nn.Module):
    """docstring for ClassName"""

    def __init__(self, ratio=1):
        super(Scale_Transfer_Layer, self).__init__()
        self.ratio = ratio
        self.scale_layer = nn.PixelShuffle(self.ratio)

    def forward(self, x):
        x = self.scale_layer(x)
        return x


class DetectionLayer(nn.Module):
    """docstring for DetectionLayer"""

    def __init__(self, num_input_features, out_channel):
        super(DetectionLayer, self).__init__()
        self.mid_channel = 256
        self.features = nn.Sequential(
            nn.Conv2d(num_input_features,  self.mid_channel,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channel, self.mid_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channel, out_channel,
                      kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.features(x)
        return out


class STDN(nn.Module):
    """docstring for STDN"""

    def __init__(self, phase, size, config=None):
        super(STDN, self).__init__()
        self.phase = phase
        self.size = size
        self.init_params(config)
        print_info('===> Constructing STDN model', ['yellow', 'bold'])
        self.construct_modules()

    def init_params(self, config=None):
        assert config is not None, 'Error:no config'
        for key, value in config.items():
            setattr(self, key, value)

    def construct_modules(self,):
        self.densenet = densenet169()

        self.scale_transfer_block = (
            nn.AvgPool2d(kernel_size=self.scale_param[
                         0], stride=self.scale_param[0]),
            nn.AvgPool2d(kernel_size=self.scale_param[
                         1], stride=self.scale_param[1]),
            nn.AvgPool2d(kernel_size=self.scale_param[
                         2], stride=self.scale_param[2], ceil_mode=True),
            Scale_Transfer_Layer(self.scale_param[3]),
            Scale_Transfer_Layer(self.scale_param[4])
        )

        loc_ = list()
        conf_ = list()

        for i, c in enumerate(self.predict_channel):
            loc_.append(DetectionLayer(c, 4 * self.anchor_number[i]))
            conf_.append(DetectionLayer(
                c, self.num_classes * self.anchor_number[i]))

        self.loc = nn.ModuleList(loc_)
        self.conf = nn.ModuleList(conf_)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        soureces = self.densenet(x)

        feat = []
        for i, s in enumerate(soureces):
            if i == 3:
                feat += [s]
            elif i < 3:
                feat += [self.scale_transfer_block[i](s)]
            else:
                feat += [self.scale_transfer_block[i - 1](s)]
        feat = reversed(feat)
        loc, conf = list(), list()
        for (x, l, c) in zip(feat, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes))  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes)
            )
        return output

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight)
                if 'bias' in m.state_dict().keys():
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print_info('Load weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print_info('Finished!')
        else:
            print_info('Sorry only .pth and .pkl files supported.')


def build_net(phase='train', size=300, config=None):
    if not phase in ['test', 'train']:
        raise ValueError("Error: Phase not recognized")

    if not size in [300, 321, 513]:
        raise NotImplementedError(
            "Error: Sorry only STDN300,STDN321 or STDN513 are supported!")

    return STDN(phase, size, config)
