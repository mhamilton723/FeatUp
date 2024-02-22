import torch


def entropy(t):
    return -(t * torch.log(t.clamp_min(.0000001))).sum(dim=[-1, -2, -3]).mean()

def total_variation(img):
    b, c, h, w = img.size()
    return ((img[:, :, 1:, :] - img[:, :, :-1, :]).square().sum() +
            (img[:, :, :, 1:] - img[:, :, :, :-1]).square().sum()) / (b * c * h * w)
