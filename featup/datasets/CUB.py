import os
from os.path import join

from PIL import Image
from torch.utils.data import Dataset


class CUB(Dataset):
    def __init__(self, root, split, transform, **kwargs):
        self.root = join(root, "CUB_200_2011")
        self.datalist = join("datalists", "CUB", f"{split}.txt")
        self.transform = transform
        image_ids = []
        image_names = []
        image_labels = []
        with open(self.datalist) as f:
            for line in f:
                info = line.strip().split()
                image_ids.append(int(info[0]))
                image_names.append(info[1])
                image_labels.append(int(info[2]))
        self.image_ids = image_ids
        self.image_names = image_names
        self.image_labels = image_labels

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root, 'images/', image_name)
        image_label = self.image_labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        batch = {
            "img": image,
            "label": image_label,
            "img_path": image_path
        }
        return batch

    def __len__(self):
        return len(self.image_ids)
