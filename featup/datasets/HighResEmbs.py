import collections
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import default_collate
from tqdm import tqdm

from util import get_dataset

torch.multiprocessing.set_sharing_strategy('file_system')


def clamp_mag(t, min_mag, max_mag):
    mags = mag(t)
    clamped_above = t * (max_mag / mags.clamp_min(.000001)).clamp_max(1.0)
    clamped_below = clamped_above * (min_mag / mags.clamp_min(.000001)).clamp_min(1.0)
    return clamped_below


def pca(image_feats_list, dim=3, fit_pca=None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return feats.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    if fit_pca is None:
        fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = torch.from_numpy(fit_pca.transform(flatten(feats)))
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca


def mag(t):
    return t.square().sum(1, keepdim=True).sqrt()


def model_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.nn.Module):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: model_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: model_collate([d[key] for d in batch]) for key in elem}
    else:
        return default_collate(batch)


class HighResEmbHelper(Dataset):
    def __init__(self,
                 root,
                 output_root,
                 dataset_name,
                 emb_name,
                 split,
                 model_type,
                 transform,
                 target_transform,
                 limit,
                 include_labels):
        self.root = root
        self.emb_dir = join(output_root, "feats", emb_name, dataset_name, split, model_type)

        self.dataset = get_dataset(
            root, dataset_name, split, transform, target_transform, include_labels=include_labels)

        if split == 'train':
            self.dataset = Subset(self.dataset, generate_subset(len(self.dataset), 5000))
            # TODO factor this limit out

        if limit is not None:
            self.dataset = Subset(self.dataset, range(0, limit))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        batch = self.dataset[item]
        output_location = join(self.emb_dir, "/".join(batch["img_path"].split("/")[-1:]).replace(".jpg", ".pth"))
        batch["model"] = torch.load(output_location, map_location="cpu")
        return batch


def load_hr_emb(image, loaded_model, target_res):
    image = image.cuda()
    if isinstance(loaded_model["model"], list):
        hr_model = loaded_model["model"][0].cuda().eval()
        unprojector = loaded_model["unprojector"][0].eval()
    else:
        hr_model = loaded_model["model"].cuda().eval()
        unprojector = loaded_model["unprojector"].eval()

    with torch.no_grad():
        original_image = F.interpolate(
            image, size=(target_res, target_res), mode='bilinear', antialias=True)
        hr_feats = hr_model(original_image)
        return unprojector(hr_feats.detach().cpu())


class HighResEmb(Dataset):
    def __init__(self,
                 root,
                 dataset_name,
                 emb_name,
                 split,
                 output_root,
                 model_type,
                 transform,
                 target_transform,
                 target_res,
                 limit,
                 include_labels,
                 ):
        self.root = root
        self.dataset = HighResEmbHelper(
            root=root,
            output_root=output_root,
            dataset_name=dataset_name,
            emb_name=emb_name,
            split=split,
            model_type=model_type,
            transform=transform,
            target_transform=target_transform,
            limit=limit,
            include_labels=include_labels)

        self.all_hr_feats = []
        self.target_res = target_res
        loader = DataLoader(self.dataset, shuffle=False, batch_size=1, num_workers=12, collate_fn=model_collate)

        for img_num, batch in enumerate(tqdm(loader, "Loading hr embeddings")):
            with torch.no_grad():
                self.all_hr_feats.append(load_hr_emb(batch["img"], batch["model"], target_res))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        batch = self.dataset.dataset[item]
        batch["hr_feat"] = self.all_hr_feats[item].squeeze(0)
        return batch


def generate_subset(n, batch):
    np.random.seed(0)
    return np.random.permutation(n)[:batch]
