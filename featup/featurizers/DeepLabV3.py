from torch import nn


class DeepLabV3Featurizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_cls_token(self, img):
        return self.model.forward(img)

    def forward(self, img, layer_num=-1):
        return self.model.backbone(img)['out']
