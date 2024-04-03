import matplotlib.pyplot as plt
from featup.util import pca, remove_axes
from featup.featurizers.maskclip.clip import tokenize
from pytorch_lightning import seed_everything
import torch


def plot_feats(image, lr, hr):
    assert len(image.shape) == len(lr.shape) == len(hr.shape) == 3
    seed_everything(0)
    [lr_feats_pca, hr_feats_pca], _ = pca([lr.unsqueeze(0), hr.unsqueeze(0)])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.permute(1,2,0).detach().cpu())
    ax[0].set_title("Image")
    ax[1].imshow(lr_feats_pca[0].permute(1,2,0).detach().cpu())
    ax[1].set_title("Original Features")
    ax[2].imshow(hr_feats_pca[0].permute(1,2,0).detach().cpu())
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
    text_feats = model.model.encode_text(text)
    text_feats /= text_feats.norm(dim=-1, keepdim=True)

    # upscale low-res features with interpolation
    input_size = image.shape[1:]
    lr_feats = torch.nn.functional.interpolate(
                lr_feats.unsqueeze(0), 
                size=input_size, 
                mode="bicubic", 
                align_corners=False
    ).squeeze(0)
    
    # normalize features
    lr_feats = lr_feats.permute(1, 2, 0)
    hr_feats = hr_feats.permute(1, 2, 0)
    lr_feats /= lr_feats.norm(dim=-1, keepdim=True)
    hr_feats /= hr_feats.norm(dim=-1, keepdim=True)

    # compute cosine similarity
    lr_sims = (lr_feats.half() @ text_feats.t()).squeeze()
    hr_sims = (hr_feats.half() @ text_feats.t()).squeeze()

    lr_sims_norm = (lr_sims - lr_sims.min()) / (lr_sims.max() - lr_sims.min())
    hr_sims_norm = (hr_sims - hr_sims.min()) / (hr_sims.max() - hr_sims.min())
    lr_heatmap = cmap(lr_sims_norm.cpu().numpy())
    hr_heatmap = cmap(hr_sims_norm.cpu().numpy())
    
    ax[0].imshow(image.permute(1,2,0).detach().cpu())
    ax[0].set_title("Image")
    ax[1].imshow(lr_heatmap)
    ax[1].set_title(f"Original Features Similarity {text_query}")
    ax[2].imshow(hr_heatmap)
    ax[2].set_title(f"Upsampled Features Similarity {text_query}")
    remove_axes(ax)
    
    return plt.show()