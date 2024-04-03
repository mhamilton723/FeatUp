import torch

def get_featurizer(name, activation_type="key", **kwargs):
    name = name.lower()
    if name == "vit":
        from .DINO import DINOFeaturizer
        patch_size = 16
        model = DINOFeaturizer("vit_small_patch16_224", patch_size, activation_type)
        dim = 384
    elif name == "midas":
        from .MIDAS import MIDASFeaturizer
        patch_size = 16
        model = MIDASFeaturizer(output_root=kwargs["output_root"])
        dim = 768
    elif name == "dino16":
        from .DINO import DINOFeaturizer
        patch_size = 16
        model = DINOFeaturizer("dino_vits16", patch_size, activation_type)
        dim = 384
    elif name == "dino8":
        from .DINO import DINOFeaturizer
        patch_size = 8
        model = DINOFeaturizer("dino_vits8", patch_size, activation_type)
        dim = 384
    elif name == "dinov2":
        from .DINOv2 import DINOv2Featurizer
        patch_size = 14
        model = DINOv2Featurizer("dinov2_vits14", patch_size, activation_type)
        dim = 384
    elif name == "clip":
        from .CLIP import CLIPFeaturizer
        patch_size = 16
        model = CLIPFeaturizer()
        dim = 512
    elif name == "maskclip":
        from .MaskCLIP import MaskCLIPFeaturizer
        patch_size = 16
        model = MaskCLIPFeaturizer()
        dim = 512
    elif name == "mae":
        from .MAE import MAEFeaturizer
        patch_size = 16
        model = MAEFeaturizer(**kwargs)
        dim = 1024
    elif name == "mocov3":
        from .MOCOv3 import MOCOv3Featurizer
        patch_size = 16
        model = MOCOv3Featurizer()
        dim = 384
    elif name == "msn":
        from .MSN import MSNFeaturizer
        patch_size = 16
        model = MSNFeaturizer()
        dim = 384
    elif name == "pixels":
        patch_size = 1
        model = lambda x: x
        dim = 3
    elif name == "resnet50":
        from .modules.resnet import resnet50
        from .ResNet import ResNetFeaturizer
        model = ResNetFeaturizer(resnet50(pretrained=True))
        patch_size = 1
        dim = 2048
    elif name == "deeplab":
        from .DeepLabV3 import DeepLabV3Featurizer
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        model = DeepLabV3Featurizer(model)
        patch_size = 1
        dim = 2048
    else:
        raise ValueError("unknown model: {}".format(name))
    return model, patch_size, dim
