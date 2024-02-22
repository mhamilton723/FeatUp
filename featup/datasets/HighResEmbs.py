import collections
import sys
from os.path import join

import featup.downsamplers
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from featup.featurizers.util import get_featurizer
from featup.layers import ChannelNorm
from featup.layers import ChannelNorm
from featup.util import norm
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
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
        state_dicts = torch.load(output_location, map_location="cpu")
        from featup.train_implicit_upsampler import get_implicit_upsampler
        from featup.util import PCAUnprojector
        model = get_implicit_upsampler(**state_dicts["model_args"])
        model.load_state_dict(state_dicts["model"])
        unp_state_dict = state_dicts["unprojector"]
        unprojector = PCAUnprojector(
            None,
            unp_state_dict["components_"].shape[0],
            device="cpu",
            original_dim=unp_state_dict["components_"].shape[1],
            **unp_state_dict
        )
        batch["model"] = {"model": model, "unprojector": unprojector}
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


def load_some_hr_feats(model_type,
                       activation_type,
                       dataset_name,
                       split,
                       emb_name,
                       root,
                       output_root,
                       input_size,
                       samples_per_batch,
                       num_batches,
                       num_workers
                       ):
    transform = T.Compose([
        T.Resize(input_size),
        T.CenterCrop(input_size),
        T.ToTensor(),
        norm
    ])

    shared_args = dict(
        root=root,
        dataset_name=dataset_name,
        emb_name=emb_name,
        output_root=output_root,
        model_type=model_type,
        transform=transform,
        target_transform=None,
        target_res=input_size,
        include_labels=False,
        limit=samples_per_batch * num_batches
    )

    def get_data(model, ds):
        loader = DataLoader(ds, batch_size=samples_per_batch, num_workers=num_workers)
        all_batches = []
        for batch in loader:
            batch["lr_feat"] = model(batch["img"].cuda()).cpu()
            all_batches.append(batch)

        big_batch = {}
        for k, t in all_batches[0].items():
            if isinstance(t, torch.Tensor):
                big_batch[k] = torch.cat([b[k] for b in all_batches], dim=0)
        del loader
        return big_batch

    with torch.no_grad():
        model, _, dim = get_featurizer(model_type, activation_type)
        model = torch.nn.Sequential(model, ChannelNorm(dim))
        model = model.cuda()
        batch = get_data(model, HighResEmb(split=split, **shared_args))
        del model

    return batch


if __name__ == "__main__":
    loaded = load_some_hr_feats(
        "vit",
        "token",
        "cocostuff",
        "train",
        "3_12_2024",
        "/pytorch-data/",
        "../../../",
        224,
        50,
        3,
        0
    )

    print(loaded)
