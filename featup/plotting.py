import matplotlib.pyplot as plt
from featup.util import pca, remove_axes
from featup.featurizers.maskclip.clip import tokenize
from pytorch_lightning import seed_everything
import torch
import torch.nn.functional as F


@torch.no_grad()
def plot_feats(image, lr, hr):
    assert len(image.shape) == len(lr.shape) == len(hr.shape) == 3
    seed_everything(0)
    [lr_feats_pca, hr_feats_pca], _ = pca([lr.unsqueeze(0), hr.unsqueeze(0)])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[0].set_title("Image")
    ax[1].imshow(lr_feats_pca[0].permute(1, 2, 0).detach().cpu())
    ax[1].set_title("Original Features")
    ax[2].imshow(hr_feats_pca[0].permute(1, 2, 0).detach().cpu())
    ax[2].set_title("Upsampled Features")
    remove_axes(ax)
    plt.show()


@torch.no_grad()
def plot_lang_heatmaps(model, image, lr_feats, hr_feats, text_query):
    assert len(image.shape) == len(lr_feats.shape) == len(hr_feats.shape) == 3
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.get_cmap("turbo")

    # encode query
    text = tokenize(text_query).to(lr_feats.device)
    text_feats = model.model.encode_text(text).squeeze().to(torch.float32)
    assert len(text_feats.shape) == 1

    lr_sims = torch.einsum(
        "chw,c->hw", F.normalize(lr_feats.to(torch.float32), dim=0), F.normalize(text_feats, dim=0))
    hr_sims = torch.einsum(
        "chw,c->hw", F.normalize(hr_feats.to(torch.float32), dim=0), F.normalize(text_feats, dim=0))

    lr_sims_norm = (lr_sims - lr_sims.min()) / (lr_sims.max() - lr_sims.min())
    hr_sims_norm = (hr_sims - hr_sims.min()) / (hr_sims.max() - hr_sims.min())
    lr_heatmap = cmap(lr_sims_norm.cpu().numpy())
    hr_heatmap = cmap(hr_sims_norm.cpu().numpy())

    ax[0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[0].set_title("Image")
    ax[1].imshow(lr_heatmap)
    ax[1].set_title(f"Original Similarity to \"{text_query}\"")
    ax[2].imshow(hr_heatmap)
    ax[2].set_title(f"Upsampled Similarity to \"{text_query}\"")
    remove_axes(ax)

    return plt.show()
