import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from timm.models.layers import get_act_layer
import numpy as np
from torch.nn.functional import interpolate
import os

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, exportable=True, hooks=None,
                  use_vit_only=False, use_readout="ignore", in_features=[96, 256, 512, 1024]):
    if backbone == "levit_384":
        pretrained = _make_pretrained_levit_384(
            use_pretrained, hooks=hooks
        )
        scratch = _make_scratch(
            [384, 512, 768], features, groups=groups, expand=expand
        )  # LeViT 384 (backbone)
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def forward_levit(pretrained, x):
    pretrained.model.forward_features(x)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]

    layer_1 = pretrained.act_postprocess1(layer_1)
    layer_2 = pretrained.act_postprocess2(layer_2)
    layer_3 = pretrained.act_postprocess3(layer_3)

    return layer_1, layer_2, layer_3


def _make_levit_backbone(
        model,
        hooks=[3, 11, 21],
        patch_grid=[14, 14]
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))

    pretrained.activations = activations

    patch_grid_size = np.array(patch_grid, dtype=int)

    pretrained.act_postprocess1 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size(patch_grid_size.tolist()))
    )
    pretrained.act_postprocess2 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((np.ceil(patch_grid_size / 2).astype(int)).tolist()))
    )
    pretrained.act_postprocess3 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((np.ceil(patch_grid_size / 4).astype(int)).tolist()))
    )

    return pretrained


class ConvTransposeNorm(nn.Sequential):
    """
    Modification of
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/levit.py: ConvNorm
    such that ConvTranspose2d is used instead of Conv2d.
    """

    def __init__(
            self, in_chs, out_chs, kernel_size=1, stride=1, pad=0, dilation=1,
            groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c',
                        nn.ConvTranspose2d(in_chs, out_chs, kernel_size, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_chs))

        nn.init.constant_(self.bn.weight, bn_weight_init)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.ConvTranspose2d(
            w.size(1), w.size(0), w.shape[2:], stride=self.c.stride,
            padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def stem_b4_transpose(in_chs, out_chs, activation):
    """
    Modification of
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/levit.py: stem_b16
    such that ConvTranspose2d is used instead of Conv2d and stem is also reduced to the half.
    """
    return nn.Sequential(
        ConvTransposeNorm(in_chs, out_chs, 3, 2, 1),
        activation(),
        ConvTransposeNorm(out_chs, out_chs // 2, 3, 2, 1),
        activation())


def _make_pretrained_levit_384(pretrained, hooks=None):
    model = timm.create_model("levit_384", pretrained=pretrained)

    hooks = [3, 11, 21] if hooks == None else hooks
    return _make_levit_backbone(
        model,
        hooks=hooks
    )


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPT(BaseModel):
    def __init__(
            self,
            head,
            features=256,
            backbone="vitb_rn50_384",
            readout="project",
            channels_last=False,
            use_bn=False,
            **kwargs
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        # For the Swin, Swin 2, LeViT and Next-ViT Transformers, the hierarchical architectures prevent setting the
        # hooks freely. Instead, the hooks have to be chosen according to the ranges specified in the comments.
        hooks = {
            "beitl16_512": [5, 11, 17, 23],
            "beitl16_384": [5, 11, 17, 23],
            "beitb16_384": [2, 5, 8, 11],
            "swin2l24_384": [1, 1, 17, 1],  # Allowed ranges: [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "swin2b24_384": [1, 1, 17, 1],  # [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "swin2t16_256": [1, 1, 5, 1],  # [0, 1], [0,  1], [ 0,  5], [ 0,  1]
            "swinl12_384": [1, 1, 17, 1],  # [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "next_vit_large_6m": [2, 6, 36, 39],  # [0, 2], [3,  6], [ 7, 36], [37, 39]
            "levit_384": [3, 11, 21],  # [0, 3], [6, 11], [14, 21]
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }[backbone]

        if "next_vit" in backbone:
            in_features = {
                "next_vit_large_6m": [96, 256, 512, 1024],
            }[backbone]
        else:
            in_features = None

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks,
            use_readout=readout,
            in_features=in_features,
        )

        self.number_layers = len(hooks) if hooks is not None else 4
        self.scratch.stem_transpose = None

        self.forward_transformer = forward_levit
        size_refinenet3 = 7
        self.scratch.stem_transpose = stem_b4_transpose(256, 128, get_act_layer("hard_swish"))

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, size_refinenet3)
        if self.number_layers >= 4:
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward_features(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layers = self.forward_transformer(self.pretrained, x)
        if self.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
        else:
            layer_1, layer_2, layer_3, layer_4 = layers

        all_feats = []
        target_size = layer_1.shape[2:]

        def prep(l):
            if target_size != l.shape[2:]:
                l = interpolate(l, size=target_size, mode="bilinear")
            return l

        all_feats.append(prep(self.scratch.layer1_rn(layer_1)))
        all_feats.append(prep(self.scratch.layer2_rn(layer_2)))
        all_feats.append(prep(self.scratch.layer3_rn(layer_3)))
        if self.number_layers >= 4:
            all_feats.append(prep(self.scratch.layer4_rn(layer_4)))
        return torch.cat([f for f in all_feats], dim=1)

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layers = self.forward_transformer(self.pretrained, x)
        if self.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
        else:
            layer_1, layer_2, layer_3, layer_4 = layers

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        if self.number_layers >= 4:
            layer_4_rn = self.scratch.layer4_rn(layer_4)

        if self.number_layers == 3:
            path_3 = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        else:
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.scratch.stem_transpose is not None:
            path_1 = self.scratch.stem_transpose(path_1)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        head_features_1 = kwargs["head_features_1"] if "head_features_1" in kwargs else features
        head_features_2 = kwargs["head_features_2"] if "head_features_2" in kwargs else 32
        kwargs.pop("head_features_1", None)
        kwargs.pop("head_features_2", None)

        head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        return super().forward(x).squeeze(dim=1)

    def forward_features(self, x):
        return super().forward_features(x).squeeze(dim=1)


class MIDASFeaturizer(nn.Module):

    def __init__(self, output_root):
        super().__init__()
        self.model = DPTDepthModel(
            path=os.path.join(output_root, 'models/dpt_levit_224.pt'),
            backbone="levit_384",
            non_negative=True,
            head_features_1=64,
            head_features_2=8,
        )

    def get_cls_token(self, img):
        return None

    def forward(self, img):
        feats = self.model.forward_features(img)
        return feats


if __name__ == "__main__":
    DPTDepthModel(
        path='../../models/dpt_levit_224.pt',
        backbone="levit_384",
        non_negative=True,
        head_features_1=64,
        head_features_2=8,
    ).cuda()

    image = Image.open("../../sample-images/car.jpg").convert("RGB")

    input_size = 224

    transform = T.Compose([
        T.Resize(input_size),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3)
    ])

    t_img = transform(image).unsqueeze(0).cuda()

    with torch.no_grad():
        prediction = model.forward(t_img)

        import matplotlib.pyplot as plt

        plt.imshow(prediction.squeeze().cpu())
        plt.show()
        print("here")
