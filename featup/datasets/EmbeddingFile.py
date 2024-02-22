import numpy as np
from torch.utils.data import Dataset


class EmbeddingFile(Dataset):
    """
    modified from: https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    uses cached directory listing if available rather than walking directory
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, file):
        super(Dataset, self).__init__()
        self.file = file
        loaded = np.load(file)
        self.feats = loaded["feats"]
        self.labels = loaded["labels"]

    def dim(self):
        return self.feats.shape[1]

    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, index):
        return self.feats[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class EmbeddingAndImage(Dataset):
    def __init__(self, file, dataset):
        super(Dataset, self).__init__()
        self.file = file
        loaded = np.load(file)
        self.feats = loaded["feats"]
        self.labels = loaded["labels"]
        self.imgs = dataset

    def dim(self):
        return self.feats.shape[1]

    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, index):
        return self.feats[index], self.labels[index], self.imgs[index]

    def __len__(self):
        return len(self.labels)
