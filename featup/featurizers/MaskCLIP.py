import torch
from torch import nn
import os

from featup.featurizers.maskclip import clip


class MaskCLIPFeaturizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load(
            "ViT-B/16",
            download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
        )
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(self, img):
        b, _, input_size_h, input_size_w = img.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.get_patch_encodings(img).to(torch.float32)
        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)


if __name__ == "__main__":
    import torchvision.transforms as T
    from PIL import Image
    from featup.util import norm, unnorm, crop_to_divisor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = Image.open("../samples/lex1.jpg")
    load_size = 224  # * 3
    transform = T.Compose([
        T.Resize(load_size, Image.BILINEAR),
        # T.CenterCrop(load_size),
        T.ToTensor(),
        lambda x: crop_to_divisor(x, 16),
        norm])

    model = MaskCLIPFeaturizer().cuda()

    results = model(transform(image).cuda().unsqueeze(0))

    print(clip.available_models())
