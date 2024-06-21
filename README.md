# Implementing Hydranets using pytorch
We will try to implement [Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations](https://arxiv.org/pdf/1809.04766.pdf)

<img width="1077" alt="image" src="https://github.com/Vishalkagade/Hydranets/assets/105672962/3b01520e-3b87-4a3a-9cfa-569afcc0a281">


  

The paper proposes a multitasking algorithm for detecting the 3 different head which will learn the featured from shared backbone.The model outputs depth, segmentation and normals map.
The mobilenet architecture is used as encoder and ligh weight refine net is used as decoder with speacial CRP block skip connections.
### Let's dive into it

Before staring to implement its is very necessary to understand which task should be learned with which task so we will have highest benifit.According to reasearch if any task learned 
with normals task, then subsequent task also gets improved.
<img width="428" alt="image" src="https://github.com/Vishalkagade/Hydranets/assets/105672962/3aec7c88-7753-4962-849e-c89e37e0ea7f">

Note: If you are not understanding something , then please refer hand written notes above to get more confused..


Lets do a boring stuff first...Loding the libraries :)

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
```
The paper proposes Mobilenetv2 encoder for real time inference, which uses depthwise and pointwise convolutions.
<img width="428" alt="image" src="https://github.com/Vishalkagade/Hydranets/assets/105672962/8a36d7e2-d355-42f2-9623-57257badd367">
So lets create the ppointwise and depthwise convolutions:
```python
def conv3x3(in_channels, out_channels, stride=1, dilation=1, groups=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=bias, groups=groups)
def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False,):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias, groups=groups)
def batchnorm(num_features):
    return nn.BatchNorm2d(num_features, affine=True, eps=1e-5, momentum=0.1)
def convbnrelu(in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels),
                             nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels))
```
Unlike mobilenetv1,mobilenetv2 used `inverted residual block`, which follow `Narrow -> wide-> wide -> Narrow` configuration(please refer pdf above).
```python
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super().__init__() # Python 3
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1) # Boolean/Condition
        self.output = nn.Sequential(convbnrelu(in_planes, intermed_planes, 1),
                                    convbnrelu(intermed_planes, intermed_planes, 3, stride=stride, groups=intermed_planes),
                                    convbnrelu(intermed_planes, out_planes, 1, act=False))

    def forward(self, x):
        #residual = x
        out = self.output(x)
        if self.residual:
            return (out + x)#+residual
        else:
            return out
```
Now combining the mobilenet:
````python
def define_mobilenet(self):
    mobilenet_config = [[1, 16, 1, 1], # expansion rate, output channels, number of repeats, stride
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
                    ]
    self.in_channels = 32 # number of input channels
    self.num_layers = len(mobilenet_config)
    self.layer1 = convbnrelu(3, self.in_channels, kernel_size=3, stride=2)
    c_layer = 2
    for t,c,n,s in (mobilenet_config):
        layers = []
        for idx in range(n):
            layers.append(InvertedResidualBlock(self.in_channels, c, expansion_factor=t, stride=s if idx == 0 else 1))
            self.in_channels = c
        setattr(self, 'layer{}'.format(c_layer), nn.Sequential(*layers)) # setattr(object, name, value)
        c_layer += 1
```
Now, paper proposes the ligh weight refinenet decoder,which used pretrained mobilnetv2 as backbone and takes input at 4 different scales i.e ´1/4, 1/8, 1/16, 1/32´

