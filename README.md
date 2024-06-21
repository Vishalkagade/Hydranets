# Implementing Hydranets using pytorch
We will try to implement [Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations](https://arxiv.org/pdf/1809.04766.pdf)

<div style="display: flex; align-items: flex-start;">
  <img src="https://github.com/Vishalkagade/Hydranets/assets/105672962/de499153-3a4e-4b5e-b5cf-8652dfa78ba1" width="500" alt="Hydranets Video">
</div>

The paper proposes a multitasking algorithm for detecting the 3 different head which will learn the featured from shared backbone.The model outputs depth, segmentation and normals map.
The mobilenet architecture is used as encoder and ligh weight refine net is used as decoder with speacial CRP block skip connections.
<div style="display: flex; justify-content: center;">
  <img src="https://github.com/Vishalkagade/Hydranets/assets/105672962/3b01520e-3b87-4a3a-9cfa-569afcc0a281" width="700" alt="Hydranets">
</div>
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
```python
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

Now, paper proposes the ligh weight refinenet decoder,which used pretrained mobilnetv2 as backbone and takes input at 4 different scales i.e `1/4, 1/8, 1/16, 1/32`

<img width="759" alt="image" src="https://github.com/Vishalkagade/Hydranets/assets/105672962/de6fc341-d2b2-41e5-a22f-276053a09002">

Author has not used `RCU blocks` as they observed that removing RCU blocks did not lead to any accuracy deterioration, and, in fact, the weights in RCU blocks almost completely
saturated.Note that RCU only dont work when we are using light weight networks,this happens due to the RCU blocks being redundantin the 1 × 1 convolution regime, as the only important goal of increasing the contextual
coverage is essentially performed by pooling layers inside CRP.

The feature extraction from backbone starts froom lower resolution and then passed to ` CRP block` and then it is fused to the upper resolution by `upsampling`.
<img width="359" alt="image" src="https://github.com/Vishalkagade/Hydranets/assets/105672962/cdea2b8a-422c-4559-8372-6a99af8d68b1">

chained residual pooling(CRP) 
```python

class CRPBlock(nn.Module):
    """CRP definition"""
    def __init__(self, in_planes, out_planes, n_stages, groups=False):
        super().__init__() #Python 3
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False, groups=in_planes if groups else 1)) #setattr(object, name, value)

        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)#getattr(object, name[, default])
            x = top + x
        return x
def _make_crp(self, in_planes, out_planes, stages, groups=False):
    layers = [CRPBlock(in_planes, out_planes,stages, groups=groups)]
    return nn.Sequential(*layers)
```
Now concatinating aall the layers with the right dimentions is the only thing remaining.So referring my beutiful sketching skills as follows:
<img width="657" alt="image" src="https://github.com/Vishalkagade/Hydranets/assets/105672962/f9660c36-8843-4f5a-97af-621c9e985f34">

```python
def define_lightweight_refinenet(self):
    ## Light-Weight RefineNet ##
    self.conv8 = conv1x1(320, 256, bias=False)
    self.conv7 = conv1x1(160, 256, bias=False)
    self.conv6 = conv1x1(96, 256, bias=False)
    self.conv5 = conv1x1(64, 256, bias=False)
    self.conv4 = conv1x1(32, 256, bias=False)
    self.conv3 = conv1x1(24, 256, bias=False)
    self.crp4 = self._make_crp(256, 256, 4, groups=False)
    self.crp3 = self._make_crp(256, 256, 4, groups=False)
    self.crp2 = self._make_crp(256, 256, 4, groups=False)
    self.crp1 = self._make_crp(256, 256, 4, groups=True)

    self.conv_adapt4 = conv1x1(256, 256, bias=False)
    self.conv_adapt3 = conv1x1(256, 256, bias=False)
    self.conv_adapt2 = conv1x1(256, 256, bias=False)

    self.pre_depth = conv1x1(256, 256, groups=256, bias=False)
    self.depth = conv3x3(256, 1, bias=True)
    self.pre_segm = conv1x1(256, 256, groups=256, bias=False)
    self.segm = conv3x3(256, self.num_classes, bias=True)
    self.relu = nn.ReLU6(inplace=True)

    if self.num_tasks == 3:
        self.pre_normal = conv1x1(256, 256, groups=256, bias=False)
        self.normal = conv3x3(256, 3, bias=True)
```


```python
def forward(self, x):
    # MOBILENET V2
    x = self.layer1(x)
    x = self.layer2(x) # x / 2
    l3 = self.layer3(x) # 24, x / 4
    l4 = self.layer4(l3) # 32, x / 8
    l5 = self.layer5(l4) # 64, x / 16
    l6 = self.layer6(l5) # 96, x / 16
    l7 = self.layer7(l6) # 160, x / 32
    l8 = self.layer8(l7) # 320, x / 32

    # LIGHT-WEIGHT REFINENET
    l8 = self.conv8(l8)
    l7 = self.conv7(l7)
    l7 = self.relu(l8 + l7)
    l7 = self.crp4(l7)
    l7 = self.conv_adapt4(l7)
    l7 = nn.Upsample(size=l6.size()[2:], mode='bilinear', align_corners=False)(l7)

    l6 = self.conv6(l6)
    l5 = self.conv5(l5)
    l5 = self.relu(l5 + l6 + l7)
    l5 = self.crp3(l5)
    l5 = self.conv_adapt3(l5)
    l5 = nn.Upsample(size=l4.size()[2:], mode='bilinear', align_corners=False)(l5)

    l4 = self.conv4(l4)
    l4 = self.relu(l5 + l4)
    l4 = self.crp2(l4)
    l4 = self.conv_adapt2(l4)
    l4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=False)(l4)

    l3 = self.conv3(l3)
    l3 = self.relu(l3 + l4)
    l3 = self.crp1(l3)

    # HEADS
    out_segm = self.pre_segm(l3)
    out_segm = self.relu(out_segm)
    out_segm = self.segm(out_segm)

    out_d = self.pre_depth(l3)
    out_d = self.relu(out_d)
    out_d = self.depth(out_d)

    if self.num_tasks == 3:
        out_n = self.pre_normal(l3)
        out_n = self.relu(out_n)
        out_n = self.normal(out_n)
        return out_segm, out_d, out_n
    else:
        return out_segm, out_d
```
Now the final step, lets create hydranet class and combine all. Hydranet class takes ´2 num of tasks and 6 num of classes as´ as an input .

```python
class HydraNet(nn.Module):
    def __init__(self):
        #super(HydraNet, self).__init__() # Python2
        super().__init__() # Python 3
        self.num_tasks = 2
        self.num_classes = 6
HydraNet.define_mobilenet = define_mobilenet
HydraNet.define_lightweight_refinenet = define_lightweight_refinenet
HydraNet.forward = forward
hydranet = HydraNet()
```
To further dive into implementation pls refer author DrSleep, a researcher named Vladimir [author work](https://github.com/DrSleep/multi-task-refinenet/blob/master/src/notebooks/ExpNYUDKITTI_joint.ipynb/)





