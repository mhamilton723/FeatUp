from PIL import Image
from torch.utils.data import Dataset


class SampleImage(Dataset):
    def __init__(self, paths, transform, **kwargs):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        batch = {
            "img": image,
            "img_path": image_path
        }
        return batch

    def __len__(self):
        return len(self.paths)
