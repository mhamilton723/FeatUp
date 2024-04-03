import clip
import torch
from torch import nn
import os

class CLIPFeaturizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load(
            "ViT-B/16",
            download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
        )
        self.model.eval()

    def get_cls_token(self, img):
        return self.model.encode_image(img).to(torch.float32)

    def forward(self, img):
        features = self.model.get_visual_features(img, include_cls=False).to(torch.float32)
        return features


if __name__ == "__main__":
    import torchvision.transforms as T
    from PIL import Image
    from shared import norm, crop_to_divisor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = Image.open("../samples/lex1.jpg")
    load_size = 224  # * 3
    transform = T.Compose([
        T.Resize(load_size, Image.BILINEAR),
        # T.CenterCrop(load_size),
        T.ToTensor(),
        lambda x: crop_to_divisor(x, 16),
        norm])

    model = CLIPFeaturizer().cuda()

    results = model(transform(image).cuda().unsqueeze(0))

    print(clip.available_models())
