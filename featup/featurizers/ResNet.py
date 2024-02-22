from torch import nn


class ResNetFeaturizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_cls_token(self, img):
        return self.model.forward(img)

    def get_layer(self, img, layer_num):
        return self.model.get_layer(img, layer_num)

    def forward(self, img, layer_num=-1):
        return self.model.get_layer(img, layer_num)
