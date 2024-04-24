import os
from torch.utils.data import Dataset
from featup.datasets.ImageNetSubset import ImageNetSubset
from featup.datasets.COCO import Coco
from featup.datasets.DAVIS import DAVIS
from featup.datasets.SampleImage import SampleImage


class SlicedDataset(Dataset):
    def __init__(self, ds, start, end):
        self.ds = ds
        self.start = max(0, start)
        self.end = min(len(ds), end)

    def __getitem__(self, index):
        if index >= self.__len__():
            raise StopIteration

        return self.ds[self.start + index]

    def __len__(self):
        return self.end - self.start



class SingleImageDataset(Dataset):
    def __init__(self, i, ds, l=None):
        self.ds = ds
        self.i = i
        self.l = len(self.ds) if l is None else l

    def __len__(self):
        return self.l

    def __getitem__(self, item):
        return self.ds[self.i]


def get_dataset(dataroot, name, split, transform, target_transform, include_labels):
    if name == 'imagenet':
        if split == 'val':
            imagenet_subset = f'datalists/val_paths_vit.txt'
        else:
            imagenet_subset = None

        return ImageNetSubset(dataroot, split, transform, target_transform,
                              include_labels=include_labels, subset=imagenet_subset)
    elif name == 'cocostuff':
        return Coco(dataroot, split, transform, target_transform, include_labels=include_labels)
    elif name.startswith('davis_'):
        return DAVIS(dataroot, name.split("_")[-1], transform)
    elif name == "sample":
        return SampleImage(
            paths=["../sample-images/bird_left.jpg",
                   "../sample-images/bird_right.jpg"],
            transform=transform
        )
    elif name == "custom":
        list_img = os.listdir(dataroot)
        return SampleImage(
            paths=[os.path.join(dataroot, path) for path in list_img],
            transform=transform
        )        
    else:
        raise ValueError(f"Unknown dataset {name}")
