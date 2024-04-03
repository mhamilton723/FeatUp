import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from featup.featurizers.modules.layers import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG_spread(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG_spread, self).__init__()
        self.features = features
        self.avgpool = AdaptiveAvgPool2d((7, 7))
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def relprop(self, R, alpha):
        x = self.classifier.relprop(R, alpha)
        x = x.reshape_as(next(reversed(self.features._modules.values())).Y)
        x = self.avgpool.relprop(x, alpha)
        x = self.features.relprop(x, alpha)
        return x

    def m_relprop(self, R, pred, alpha):
        x = self.classifier.m_relprop(R, pred, alpha)
        if torch.is_tensor(x) == False:
            for i in range(len(x)):
                x[i] = x[i].reshape_as(next(reversed(self.features._modules.values())).Y)
        else:
            x = x.reshape_as(next(reversed(self.features._modules.values())).Y)
        x = self.avgpool.m_relprop(x, pred, alpha)
        x = self.features.m_relprop(x, pred, alpha)
        return x

    def RAP_relprop(self, R):
        x1 = self.classifier.RAP_relprop(R)
        if torch.is_tensor(x1) == False:
            for i in range(len(x1)):
                x1[i] = x1[i].reshape_as(next(reversed(self.features._modules.values())).Y)
        else:
            x1 = x1.reshape_as(next(reversed(self.features._modules.values())).Y)
        x1 = self.avgpool.RAP_relprop(x1)
        x1 = self.features.RAP_relprop(x1)
        return x1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = AdaptiveAvgPool2d((7, 7))
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, num_classes),
        )
        self.num_classes = num_classes
        if init_weights:
            self._initialize_weights()

    def CLRP(self, x, maxindex = [None]):
        if maxindex == [None]:
            maxindex = torch.argmax(x, dim=1)
        R = torch.ones(x.shape, x.device)
        R /= -self.num_classes
        for i in range(R.size(0)):
            R[i, maxindex[i]] = 1
        return R

    def upsample(self, source, guidance_unscaled, upsampler, scale):
        _, _, H, W = source.shape
        guidance = F.interpolate(guidance_unscaled, size=(H * scale, W * scale), mode='bilinear')
        return upsampler(source, guidance)

    def forward(self, x,mode='output', target_class = [None], upsampler=None, scale=1):
        inp = copy.deepcopy(x)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if mode.lstrip('-').isnumeric():
                if int(mode) == i:
                    target_layer = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if mode == 'output':
            return x

        R = self.CLRP(x, target_class)
        R = self.classifier.relprop(R)
        R = R.reshape_as(next(reversed(self.features._modules.values())).Y)
        R = self.avgpool.relprop(R)

        for i in range(len(self.features)-1, int(mode), -1):
            R = self.features[i].relprop(R)

        if upsampler is not None:
            target_layer = self.upsample(target_layer, inp, upsampler, scale)

        r_weight = torch.mean(R, dim=(2, 3), keepdim=True)
        r_cam = target_layer * r_weight
        r_cam = torch.sum(r_cam, dim=(1), keepdim=True)
        return r_cam, x



    def relprop(self, R, alpha, flag=-1):
        x = self.classifier.relprop(R, alpha)
        x = x.reshape_as(next(reversed(self.features._modules.values())).Y)
        x = self.avgpool.relprop(x, alpha)
        # x = self.features.relprop(x, alpha)
        for i in range(43, flag, -1):
            x = self.features[i].relprop(x, alpha)
        return x

    def m_relprop(self, R, pred, alpha):
        x = self.classifier.m_relprop(R, pred, alpha)
        if torch.is_tensor(x) == False:
            for i in range(len(x)):
                x[i] = x[i].reshape_as(next(reversed(self.features._modules.values())).Y)
        else:
            x = x.reshape_as(next(reversed(self.features._modules.values())).Y)
        x = self.avgpool.m_relprop(x, pred, alpha)
        x = self.features.m_relprop(x, pred, alpha)
        return x

    def RAP_relprop(self, R):
        x1 = self.classifier.RAP_relprop(R)
        if torch.is_tensor(x1) == False:
            for i in range(len(x1)):
                x1[i] = x1[i].reshape_as(next(reversed(self.features._modules.values())).Y)
        else:
            x1 = x1.reshape_as(next(reversed(self.features._modules.values())).Y)
        x1 = self.avgpool.RAP_relprop(x1)
        x1 = self.features.RAP_relprop(x1)

        return x1
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), ReLU(inplace=True)]
            else:
                layers += [conv2d, ReLU(inplace=True)]
            in_channels = v

    return Sequential(*layers)

def make_layers_list(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), ReLU(inplace=True)]
            else:
                layers += [conv2d, ReLU(inplace=True)]
            in_channels = v
    return layers


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

def vgg16_spread(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_spread(make_layers_list(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model
