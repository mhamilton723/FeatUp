import gc
import os

import hydra
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from os.path import join

from featup.datasets.JitteredImage import apply_jitter, sample_transform
from featup.datasets.util import get_dataset, SingleImageDataset
from featup.downsamplers import SimpleDownsampler, AttentionDownsampler
from featup.featurizers.util import get_featurizer
from featup.layers import ChannelNorm
from featup.losses import TVLoss, SampledCRFLoss, entropy
from featup.upsamplers import get_upsampler
from featup.util import pca, RollingAvg, unnorm, norm, prep_image

torch.multiprocessing.set_sharing_strategy('file_system')


class ScaleNet(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = torch.nn.Conv2d(dim, 1, 1)
        with torch.no_grad():
            self.net.weight.copy_(self.net.weight * .1)
            self.net.bias.copy_(self.net.bias * .1)

    def forward(self, x):
        return torch.exp(self.net(x) + .1).clamp_min(.0001)


class JBUFeatUp(pl.LightningModule):
    def __init__(self,
                 model_type,
                 activation_type,
                 n_jitters,
                 max_pad,
                 max_zoom,
                 kernel_size,
                 final_size,
                 lr,
                 random_projection,
                 predicted_uncertainty,
                 crf_weight,
                 filter_ent_weight,
                 tv_weight,
                 upsampler,
                 downsampler,
                 chkpt_dir,
                 ):
        super().__init__()
        self.model_type = model_type
        self.activation_type = activation_type
        self.n_jitters = n_jitters
        self.max_pad = max_pad
        self.max_zoom = max_zoom
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.lr = lr
        self.random_projection = random_projection
        self.predicted_uncertainty = predicted_uncertainty
        self.crf_weight = crf_weight
        self.filter_ent_weight = filter_ent_weight
        self.tv_weight = tv_weight
        self.chkpt_dir = chkpt_dir

        self.model, self.patch_size, self.dim = get_featurizer(model_type, activation_type, num_classes=1000)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = torch.nn.Sequential(self.model, ChannelNorm(self.dim))
        self.upsampler = get_upsampler(upsampler, self.dim)

        if downsampler == 'simple':
            self.downsampler = SimpleDownsampler(self.kernel_size, self.final_size)
        elif downsampler == 'attention':
            self.downsampler = AttentionDownsampler(self.dim, self.kernel_size, self.final_size, blur_attn=True)
        else:
            raise ValueError(f"Unknown downsampler {downsampler}")

        if self.predicted_uncertainty:
            self.scale_net = ScaleNet(self.dim)

        self.avg = RollingAvg(20)

        self.crf = SampledCRFLoss(
            alpha=.1,
            beta=.15,
            gamma=.005,
            w1=10.0,
            w2=3.0,
            shift=0.00,
            n_samples=1000)
        self.tv = TVLoss()

        self.automatic_optimization = False

    def forward(self, x):
        return self.upsampler(self.model(x))

    def project(self, feats, proj):
        if proj is None:
            return feats
        else:
            return torch.einsum("bchw,bcd->bdhw", feats, proj)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        with torch.no_grad():
            if type(batch) == dict:
                img = batch['img']
            else:
                img, _ = batch
            lr_feats = self.model(img)

        full_rec_loss = 0.0
        full_crf_loss = 0.0
        full_entropy_loss = 0.0
        full_tv_loss = 0.0
        full_total_loss = 0.0
        for i in range(self.n_jitters):
            hr_feats = self.upsampler(lr_feats, img)

            if hr_feats.shape[2] != img.shape[2]:
                hr_feats = torch.nn.functional.interpolate(hr_feats, img.shape[2:], mode="bilinear")

            with torch.no_grad():
                transform_params = sample_transform(
                    True, self.max_pad, self.max_zoom, img.shape[2], img.shape[3])
                jit_img = apply_jitter(img, self.max_pad, transform_params)
                lr_jit_feats = self.model(jit_img)

            if self.random_projection is not None:
                proj = torch.randn(lr_feats.shape[0],
                                   lr_feats.shape[1],
                                   self.random_projection, device=lr_feats.device)
                proj /= proj.square().sum(1, keepdim=True).sqrt()
            else:
                proj = None

            hr_jit_feats = apply_jitter(hr_feats, self.max_pad, transform_params)
            proj_hr_feats = self.project(hr_jit_feats, proj)

            down_jit_feats = self.project(self.downsampler(hr_jit_feats, jit_img), proj)

            if self.predicted_uncertainty:
                scales = self.scale_net(lr_jit_feats)
                scale_factor = (1 / (2 * scales ** 2))
                mse = (down_jit_feats - self.project(lr_jit_feats, proj)).square()
                rec_loss = (scale_factor * mse + scales.log()).mean() / self.n_jitters
            else:
                rec_loss = (self.project(lr_jit_feats, proj) - down_jit_feats).square().mean() / self.n_jitters

            full_rec_loss += rec_loss.item()

            if self.crf_weight > 0 and i == 0:
                crf_loss = self.crf(img, proj_hr_feats)
                full_crf_loss += crf_loss.item()
            else:
                crf_loss = 0.0

            if self.filter_ent_weight > 0.0:
                entropy_loss = entropy(self.downsampler.get_kernel())
                full_entropy_loss += entropy_loss.item()
            else:
                entropy_loss = 0

            if self.tv_weight > 0 and i == 0:
                tv_loss = self.tv(proj_hr_feats.square().sum(1, keepdim=True))
                full_tv_loss += tv_loss.item()
            else:
                tv_loss = 0.0

            loss = rec_loss + self.crf_weight * crf_loss + self.tv_weight * tv_loss - self.filter_ent_weight * entropy_loss
            full_total_loss += loss.item()
            self.manual_backward(loss)

        self.avg.add("loss/crf", full_crf_loss)
        self.avg.add("loss/ent", full_entropy_loss)
        self.avg.add("loss/tv", full_tv_loss)
        self.avg.add("loss/rec", full_rec_loss)
        self.avg.add("loss/total", full_total_loss)

        if self.global_step % 100 == 0:
            self.trainer.save_checkpoint(self.chkpt_dir[:-5] + '/' + self.chkpt_dir[:-5] + f'_{self.global_step}.ckpt')

        self.avg.logall(self.log)
        if self.global_step < 10:
            self.clip_gradients(opt, gradient_clip_val=.0001, gradient_clip_algorithm="norm")

        opt.step()

        return None

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.trainer.is_global_zero and batch_idx == 0:

                if type(batch) == dict:
                    img = batch['img']
                else:
                    img, _ = batch
                lr_feats = self.model(img)

                hr_feats = self.upsampler(lr_feats, img)

                if hr_feats.shape[2] != img.shape[2]:
                    hr_feats = torch.nn.functional.interpolate(hr_feats, img.shape[2:], mode="bilinear")

                transform_params = sample_transform(
                    True, self.max_pad, self.max_zoom, img.shape[2], img.shape[3])
                jit_img = apply_jitter(img, self.max_pad, transform_params)
                lr_jit_feats = self.model(jit_img)

                if self.random_projection is not None:
                    proj = torch.randn(lr_feats.shape[0],
                                       lr_feats.shape[1],
                                       self.random_projection, device=lr_feats.device)
                    proj /= proj.square().sum(1, keepdim=True).sqrt()
                else:
                    proj = None

                scales = self.scale_net(lr_jit_feats)

                writer = self.logger.experiment

                hr_jit_feats = apply_jitter(hr_feats, self.max_pad, transform_params)
                down_jit_feats = self.downsampler(hr_jit_feats, jit_img)

                [red_lr_feats], fit_pca = pca([lr_feats[0].unsqueeze(0)])
                [red_hr_feats], _ = pca([hr_feats[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_lr_jit_feats], _ = pca([lr_jit_feats[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_hr_jit_feats], _ = pca([hr_jit_feats[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_down_jit_feats], _ = pca([down_jit_feats[0].unsqueeze(0)], fit_pca=fit_pca)

                writer.add_image("viz/image", unnorm(img[0].unsqueeze(0))[0], self.global_step)
                writer.add_image("viz/lr_feats", red_lr_feats[0], self.global_step)
                writer.add_image("viz/hr_feats", red_hr_feats[0], self.global_step)
                writer.add_image("jit_viz/jit_image", unnorm(jit_img[0].unsqueeze(0))[0], self.global_step)
                writer.add_image("jit_viz/lr_jit_feats", red_lr_jit_feats[0], self.global_step)
                writer.add_image("jit_viz/hr_jit_feats", red_hr_jit_feats[0], self.global_step)
                writer.add_image("jit_viz/down_jit_feats", red_down_jit_feats[0], self.global_step)

                norm_scales = scales[0]
                norm_scales /= scales.max()
                writer.add_image("scales", norm_scales, self.global_step)
                writer.add_histogram("scales hist", scales, self.global_step)

                if isinstance(self.downsampler, SimpleDownsampler):
                    writer.add_image(
                        "down/filter",
                        prep_image(self.downsampler.get_kernel().squeeze(), subtract_min=False),
                        self.global_step)

                if isinstance(self.downsampler, AttentionDownsampler):
                    writer.add_image(
                        "down/att",
                        prep_image(self.downsampler.forward_attention(hr_feats, None)[0]),
                        self.global_step)
                    writer.add_image(
                        "down/w",
                        prep_image(self.downsampler.w.clone().squeeze()),
                        self.global_step)
                    writer.add_image(
                        "down/b",
                        prep_image(self.downsampler.b.clone().squeeze()),
                        self.global_step)

                writer.flush()

    def configure_optimizers(self):
        all_params = []
        all_params.extend(list(self.downsampler.parameters()))
        all_params.extend(list(self.upsampler.parameters()))

        if self.predicted_uncertainty:
            all_params.extend(list(self.scale_net.parameters()))

        return torch.optim.NAdam(all_params, lr=self.lr)


@hydra.main(config_path="configs", config_name="jbu_upsampler.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.output_root)
    seed_everything(seed=0, workers=True)

    load_size = 224

    if cfg.model_type == "dinov2":
        final_size = 16
        kernel_size = 14
    else:
        final_size = 14
        kernel_size = 16

    name = (f"{cfg.model_type}_{cfg.upsampler_type}_"
            f"{cfg.dataset}_{cfg.downsampler_type}_"
            f"crf_{cfg.crf_weight}_tv_{cfg.tv_weight}"
            f"_ent_{cfg.filter_ent_weight}")

    log_dir = join(cfg.output_root, f"logs/jbu/{name}")
    chkpt_dir = join(cfg.output_root, f"checkpoints/jbu/{name}.ckpt")
    os.makedirs(log_dir, exist_ok=True)

    model = JBUFeatUp(
        model_type=cfg.model_type,
        activation_type=cfg.activation_type,
        n_jitters=cfg.n_jitters,
        max_pad=cfg.max_pad,
        max_zoom=cfg.max_zoom,
        kernel_size=kernel_size,
        final_size=final_size,
        lr=cfg.lr,
        random_projection=cfg.random_projection,
        predicted_uncertainty=cfg.outlier_detection,
        crf_weight=cfg.crf_weight,
        filter_ent_weight=cfg.filter_ent_weight,
        tv_weight=cfg.tv_weight,
        upsampler=cfg.upsampler_type,
        downsampler=cfg.downsampler_type,
        chkpt_dir=chkpt_dir
    )

    transform = T.Compose([
        T.Resize(load_size, InterpolationMode.BILINEAR),
        T.CenterCrop(load_size),
        T.ToTensor(),
        norm])

    dataset = get_dataset(
        cfg.pytorch_data_dir,
        cfg.dataset,
        "train",
        transform=transform,
        target_transform=None,
        include_labels=False)

    loader = DataLoader(
        dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(
        SingleImageDataset(0, dataset, 1), 1, shuffle=False, num_workers=cfg.num_workers)

    tb_logger = TensorBoardLogger(log_dir, default_hp_metric=False)
    callbacks = [ModelCheckpoint(chkpt_dir[:-5], every_n_epochs=1)]

    trainer = Trainer(
        accelerator='gpu',
        strategy="ddp",
        devices=cfg.num_gpus,
        max_epochs=cfg.epochs,
        logger=tb_logger,
        val_check_interval=100,
        log_every_n_steps=10,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
    )

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    trainer.fit(model, loader, val_loader)
    trainer.save_checkpoint(chkpt_dir)


if __name__ == "__main__":
    my_app()
