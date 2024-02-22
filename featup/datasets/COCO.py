import random
from os.path import join

import numpy as np
import torch
import torch.multiprocessing
from PIL import Image
from torch.utils.data import Dataset


def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


class Coco(Dataset):
    def __init__(self,
                 root,
                 split,
                 transform,
                 target_transform,
                 include_labels=True,
                 coarse_labels=False,
                 exclude_things=False,
                 subset=None):
        super(Coco, self).__init__()
        self.split = split
        self.root = join(root, "cocostuff")
        self.coarse_labels = coarse_labels
        self.transform = transform
        self.label_transform = target_transform
        self.subset = subset
        self.exclude_things = exclude_things
        self.include_labels = include_labels

        if self.subset is None:
            self.image_list = "Coco164kFull_Stuff_Coarse.txt"
        elif self.subset == 6:  # IIC Coarse
            self.image_list = "Coco164kFew_Stuff_6.txt"
        elif self.subset == 7:  # IIC Fine
            self.image_list = "Coco164kFull_Stuff_Coarse_7.txt"

        assert self.split in ["train", "val", "train+val"]
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"]
        }

        self.image_files = []
        self.label_files = []
        for split_dir in split_dirs[self.split]:
            with open(join(self.root, "curated", split_dir, self.image_list), "r") as f:
                img_ids = [fn.rstrip() for fn in f.readlines()]
                for img_id in img_ids:
                    self.image_files.append(join(self.root, "images", split_dir, img_id + ".jpg"))
                    self.label_files.append(join(self.root, "annotations", split_dir, img_id + ".png"))

        self.fine_to_coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

        self._label_names = [
            "ground-stuff",
            "plant-stuff",
            "sky-stuff",
        ]
        self.cocostuff3_coarse_classes = [23, 22, 21]
        self.first_stuff_index = 12

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        seed = np.random.randint(2147483647)
        batch = {}

        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(Image.open(image_path).convert("RGB"))
        batch["img"] = img
        batch["img_path"] = image_path

        if self.include_labels:
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.label_transform(Image.open(label_path)).squeeze(0)
            label[label == 255] = -1  # to be consistent with 10k
            coarse_label = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                coarse_label[label == fine] = coarse
            coarse_label[label == -1] = -1

            if self.coarse_labels:
                coarser_labels = -torch.ones_like(label)
                for i, c in enumerate(self.cocostuff3_coarse_classes):
                    coarser_labels[coarse_label == c] = i
                batch["label"] = coarser_labels
            else:
                if self.exclude_things:
                    batch["label"] = coarse_label - self.first_stuff_index
                else:
                    batch["label"] = coarse_label

        return batch

    @staticmethod
    def colorize_label(label):
        cmap = create_pascal_label_colormap()
        return cmap[label.cpu()].astype(np.uint8)
