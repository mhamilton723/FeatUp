# hubconf.py
import torch
from featup.featurizers.util import get_featurizer
from featup.layers import ChannelNorm
from featup.upsamplers import get_upsampler
from torch.nn import Module

dependencies = ['torch', 'torchvision', 'PIL', 'featup']  # List any dependencies here

class UpsampledBackbone(Module):

    def __init__(self, model_name):
        super().__init__()
        model, patch_size, self.dim = get_featurizer(model_name, "token", num_classes=1000)
        self.model = torch.nn.Sequential(model, ChannelNorm(self.dim))
        self.upsampler = get_upsampler("jbu_stack", self.dim)

    def forward(self, image):
        return self.upsampler(self.model(image), image)

def _load_backbone(pretrained, model_name):
    """
    The function that will be called by Torch Hub users to instantiate your model.
    Args:
        pretrained (bool): If True, returns a model pre-loaded with weights.
    Returns:
        An instance of your model.
    """
    model = UpsampledBackbone(model_name)
    if pretrained:
        # Define how you load your pretrained weights here
        # For example:
        checkpoint_url = f"https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/{model_name}_jbu_stack_cocostuff.ckpt"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if "scale_net" not in k and "downsampler" not in k}
        model.load_state_dict(state_dict, strict=False)
    return model

def vit(pretrained=True):
    return _load_backbone(pretrained, "vit")

def dino16(pretrained=True):
    return _load_backbone(pretrained, "dino16")

def clip(pretrained=True):
    return _load_backbone(pretrained, "clip")

def dinov2(pretrained=True):
    return _load_backbone(pretrained, "dinov2")

def resnet50(pretrained=True):
    return _load_backbone(pretrained, "resnet50")
