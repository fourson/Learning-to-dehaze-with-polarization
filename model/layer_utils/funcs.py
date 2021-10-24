import functools
import math

import torch
import torch.nn as nn
from torchvision import models
import numpy as np

VGG19_FEATURES = models.vgg19(pretrained=True).features
CONV3_3_IN_VGG_19 = VGG19_FEATURES[0:15].cuda()
VGG19_0to8 = VGG19_FEATURES[0:9].cuda()
VGG19_9to13 = VGG19_FEATURES[9:14].cuda()
VGG19_14to22 = VGG19_FEATURES[14:23].cuda()
VGG19_23to31 = VGG19_FEATURES[23:32].cuda()


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1, 0.02)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


