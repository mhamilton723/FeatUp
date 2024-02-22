import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import CARAFEPack
from timm.models.layers import trunc_normal_

from adaptive_conv_cuda.adaptive_conv import AdaptiveConv
from layers import ImplicitFeaturizer
from layers import id_conv


def build_spatial_kernel(sigma_spatials, radius, device):
    dist_range = torch.arange(radius * 2 + 1, device=device) - radius
    x, y = torch.meshgrid(dist_range, dist_range)
    num = (x ** 2 + y ** 2)
    denom = 2 * sigma_spatials ** 2
    denom = denom.squeeze()
    return torch.exp(-num / denom)


def setup_sigma(sigma):
    if isinstance(sigma, numbers.Number):
        return torch.tensor([sigma])
    elif isinstance(sigma, torch.Tensor):
        return sigma
    return sigma


def jbu(source, guidance, radius, sigma_spatial, sigma_range, epsilon):
    GB, GC, GH, GW = guidance.shape
    SB, SC, SH, SQ = source.shape

    assert (SB == GB)

    diameter = radius * 2 + 1

    sigma_spatial = setup_sigma(sigma_spatial).to(source.device)
    sigma_range = setup_sigma(sigma_range if sigma_range is not None else torch.std(guidance, dim=(0, 1, 2, 3))) \
        .to(source.device)

    # create high-res copy of low-res source to access floating-point coordinates
    hr_source = torch.nn.Upsample((GH, GW), mode='bilinear', align_corners=False)(source)

    guidance = F.normalize(guidance, dim=1)
    guidance_padded = F.pad(guidance, pad=[radius] * 4, mode='reflect')
    hr_source_padded = F.pad(hr_source, pad=[radius] * 4, mode='reflect')

    kernel_spatial = build_spatial_kernel(sigma_spatial, radius, source.device).to(source.device) \
        .reshape(1, diameter * diameter, 1, 1)

    range_queries = torch.nn.Unfold(diameter)(guidance_padded) \
        .reshape((GB, GC, diameter * diameter, GH, GW)) \
        .permute(0, 1, 3, 4, 2)

    if GC == 1:
        range_kernel = 1 - torch.einsum("bchwp,bchw->bphw", range_queries, guidance)
        range_kernel -= torch.amin(range_kernel, dim=(1, 2, 3), keepdim=True)
        range_kernel /= torch.amax(range_kernel, dim=(1, 2, 3), keepdim=True)
    else:
        range_kernel = 2 - torch.einsum("bchwp,bchw->bphw", range_queries, guidance)

    range_kernel = torch.exp(-range_kernel ** 2 / (2 * sigma_range ** 2).squeeze())

    combined_kernel = range_kernel * kernel_spatial
    combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(epsilon)

    combined_kernel = combined_kernel.permute(0, 2, 3, 1).view(GB, GH, GW, diameter, diameter)
    upsampled = AdaptiveConv.apply(hr_source_padded, combined_kernel)  # (B C, H+Pad, W+Pad) x (B, H, W, KH, KW) -> BCHW
    return upsampled


class JBU(torch.nn.Module):

    def __init__(self, scale=2, radius=3, sigma_spatial=4.0, sigma_range=.25, epsilon=1e-8):
        super().__init__()
        self.scale = scale
        self.radius = radius
        self.sigma_spatial = nn.Parameter(torch.tensor(sigma_spatial))
        self.sigma_range = nn.Parameter(torch.tensor(sigma_range))
        self.epsilon = epsilon

    def _forward(self, source, guidance):
        B, C, H, W = source.shape
        guidance = F.interpolate(guidance, (H * self.scale, W * self.scale), mode="bilinear", antialias=True)

        return jbu(source,
                   guidance,
                   radius=self.radius,
                   sigma_spatial=self.sigma_spatial.abs(),
                   sigma_range=self.sigma_range.abs(),
                   epsilon=self.epsilon)

    def forward(self, source, guidance):
        from adaptive_conv_cuda.adaptive_conv import AdaptiveConv

        GB, GC, GH, GW = guidance.shape
        SB, SC, SH, SQ = source.shape

        sigma_spatial = self.sigma_spatial.abs(),
        sigma_range = self.sigma_range.abs(),

        assert (SB == GB)

        diameter = self.radius * 2 + 1

        sigma_spatial = setup_sigma(sigma_spatial).to(source.device)
        sigma_range = setup_sigma(sigma_range if sigma_range is not None else torch.std(guidance, dim=(0, 1, 2, 3))) \
            .to(source.device)

        # create high-res copy of low-res source to access floating-point coordinates
        hr_source = torch.nn.Upsample((GH, GW), mode='bilinear', align_corners=False)(source)

        guidance = F.normalize(guidance, dim=1)
        guidance_padded = F.pad(guidance, pad=[self.radius] * 4, mode='reflect')
        hr_source_padded = F.pad(hr_source, pad=[self.radius] * 4, mode='reflect')

        kernel_spatial = build_spatial_kernel(sigma_spatial, self.radius, source.device).to(source.device) \
            .reshape(1, diameter * diameter, 1, 1)

        range_queries = torch.nn.Unfold(diameter)(guidance_padded) \
            .reshape((GB, GC, diameter * diameter, GH, GW)) \
            .permute(0, 1, 3, 4, 2)

        if GC == 1:
            range_kernel = 1 - torch.einsum("bchwp,bchw->bphw", range_queries, guidance)
            range_kernel -= torch.amin(range_kernel, dim=(1, 2, 3), keepdim=True)
            range_kernel /= torch.amax(range_kernel, dim=(1, 2, 3), keepdim=True)
        else:
            range_kernel = 2 - torch.einsum("bchwp,bchw->bphw", range_queries, guidance)

        range_kernel = torch.exp(-range_kernel ** 2 / (2 * sigma_range ** 2).squeeze())

        combined_kernel = range_kernel * kernel_spatial
        combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(1e-7)

        combined_kernel = combined_kernel.permute(0, 2, 3, 1).view(GB, GH, GW, diameter, diameter)
        upsampled = AdaptiveConv.apply(hr_source_padded, combined_kernel)
        # (B C, H+Pad, W+Pad) x (B, H, W, KH, KW) -> BCHW
        return upsampled


class SimpleImplicitFeaturizer(torch.nn.Module):

    def __init__(self, n_freqs=20):
        super().__init__()
        self.n_freqs = n_freqs
        self.dim_multiplier = 2

    def forward(self, original_image):
        b, c, h, w = original_image.shape
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.cat([t.unsqueeze(0) for t in torch.meshgrid([grid_h, grid_w])]).unsqueeze(0)
        feats = torch.broadcast_to(feats, (b, feats.shape[1], h, w))

        feat_list = [feats]
        feats = torch.cat(feat_list, dim=1).unsqueeze(1)
        freqs = torch.exp(torch.linspace(-2, 10, self.n_freqs, device=original_image.device)) \
            .reshape(1, self.n_freqs, 1, 1, 1)
        feats = (feats * freqs)

        feats = feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)

        all_feats = [torch.sin(feats), torch.cos(feats), original_image]

        return torch.cat(all_feats, dim=1)


class IFA(torch.nn.Module):

    def __init__(self, feat_dim, num_scales=20):
        super().__init__()
        self.scales = 2 * torch.exp(torch.tensor(torch.arange(1, num_scales + 1)))
        self.feat_dim = feat_dim
        self.sin_feats = SimpleImplicitFeaturizer()
        self.mlp = nn.Sequential(
            nn.Conv2d(feat_dim + (num_scales * 4) + 2, feat_dim, 1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(),
            nn.Conv2d(feat_dim, feat_dim, 1),
            # nn.BatchNorm2d(feat_dim),
            # nn.ReLU(),
        )

    def forward(self, source, guidance):
        b, c, h, w = source.shape
        up_source = F.interpolate(source, (h * 2, w * 2), mode="nearest")
        assert h == w
        lr_cord = torch.linspace(0, h, steps=h, device=source.device)
        hr_cord = torch.linspace(0, h, steps=2 * h, device=source.device)
        lr_coords = torch.cat([x.unsqueeze(0) for x in torch.meshgrid(lr_cord, lr_cord)], dim=0).unsqueeze(0)
        hr_coords = torch.cat([x.unsqueeze(0) for x in torch.meshgrid(hr_cord, hr_cord)], dim=0).unsqueeze(0)
        up_lr_coords = F.interpolate(lr_coords, (h * 2, w * 2), mode="nearest")
        coord_diff = up_lr_coords - hr_coords
        coord_diff_feats = self.sin_feats(coord_diff)
        c2 = coord_diff_feats.shape[1]
        bcast_coord_feats = torch.broadcast_to(coord_diff_feats, (b, c2, h * 2, w * 2))
        return self.mlp(torch.cat([up_source, bcast_coord_feats], dim=1))  # + up_source


class JBULearnedRange(torch.nn.Module):

    def __init__(self, guidance_dim, feat_dim, key_dim, scale=2, radius=3):
        super().__init__()
        self.scale = scale
        self.radius = radius
        self.diameter = self.radius * 2 + 1

        self.guidance_dim = guidance_dim
        self.key_dim = key_dim
        self.feat_dim = feat_dim

        self.range_temp = nn.Parameter(torch.tensor(0.0))
        self.range_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim, key_dim, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(key_dim, key_dim, 1, 1),
        )

        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim + self.diameter ** 2, self.diameter ** 2, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(self.diameter ** 2, self.diameter ** 2, 1, 1),
        )

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))

    def get_range_kernel(self, x):
        GB, GC, GH, GW = x.shape
        proj_x = self.range_proj(x)
        proj_x_padded = F.pad(proj_x, pad=[self.radius] * 4, mode='reflect')
        queries = torch.nn.Unfold(self.diameter)(proj_x_padded) \
            .reshape((GB, self.key_dim, self.diameter * self.diameter, GH, GW)) \
            .permute(0, 1, 3, 4, 2)
        pos_temp = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
        return F.softmax(pos_temp * torch.einsum("bchwp,bchw->bphw", queries, proj_x), dim=1)

    def get_spatial_kernel(self):
        dist_range = torch.linspace(-1, 1, self.diameter, device=self.sigma_spatial.device)
        x, y = torch.meshgrid(dist_range, dist_range)
        patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        return torch.exp(- patch.square().sum(0) / (2 * self.sigma_spatial ** 2)) \
            .reshape(1, self.diameter * self.diameter, 1, 1)

    def forward(self, source, guidance):
        GB, GC, GH, GW = guidance.shape
        SB, SC, SH, SQ = source.shape
        assert (SB == GB)

        spatial_kernel = self.get_spatial_kernel()
        range_kernel = self.get_range_kernel(guidance)

        combined_kernel = range_kernel * spatial_kernel
        combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(1e-7)

        combined_kernel += .1 * self.fixup_proj(torch.cat([combined_kernel, guidance], dim=1))
        combined_kernel = combined_kernel.permute(0, 2, 3, 1) \
            .reshape(GB, GH, GW, self.diameter, self.diameter)

        hr_source = torch.nn.Upsample((GH, GW), mode='bicubic', align_corners=False)(source)
        hr_source_padded = F.pad(hr_source, pad=[self.radius] * 4, mode='reflect')

        # (B C, H+Pad, W+Pad) x (B, H, W, KH, KW) -> BCHW
        return AdaptiveConv.apply(hr_source_padded, combined_kernel)


class JBULearnedRangeNoMLP(torch.nn.Module):

    def __init__(self, guidance_dim, feat_dim, scale=2, radius=3):
        super().__init__()
        self.scale = scale
        self.radius = radius
        self.diameter = self.radius * 2 + 1

        self.guidance_dim = guidance_dim
        self.feat_dim = feat_dim

        self.range_temp = nn.Parameter(torch.tensor(0.0))

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))

    def get_range_kernel(self, x):
        GB, GC, GH, GW = x.shape
        proj_x = x
        proj_x_padded = F.pad(proj_x, pad=[self.radius] * 4, mode='reflect')
        queries = torch.nn.Unfold(self.diameter)(proj_x_padded) \
            .reshape((GB, GC, self.diameter * self.diameter, GH, GW)) \
            .permute(0, 1, 3, 4, 2)
        pos_temp = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
        return F.softmax(pos_temp * torch.einsum("bchwp,bchw->bphw", queries, proj_x), dim=1)

    def get_spatial_kernel(self):
        dist_range = torch.linspace(-1, 1, self.diameter, device=self.sigma_spatial.device)
        x, y = torch.meshgrid(dist_range, dist_range)
        patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        return torch.exp(- patch.square().sum(0) / (2 * self.sigma_spatial ** 2)) \
            .reshape(1, self.diameter * self.diameter, 1, 1)

    def forward(self, source, guidance):
        GB, GC, GH, GW = guidance.shape
        SB, SC, SH, SQ = source.shape
        assert (SB == GB)

        spatial_kernel = self.get_spatial_kernel()
        range_kernel = self.get_range_kernel(guidance)

        combined_kernel = range_kernel * spatial_kernel
        combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(1e-7)

        combined_kernel = combined_kernel.permute(0, 2, 3, 1) \
            .reshape(GB, GH, GW, self.diameter, self.diameter)

        hr_source = torch.nn.Upsample((GH, GW), mode='bicubic', align_corners=False)(source)
        hr_source_padded = F.pad(hr_source, pad=[self.radius] * 4, mode='reflect')

        # (B C, H+Pad, W+Pad) x (B, H, W, KH, KW) -> BCHW
        return AdaptiveConv.apply(hr_source_padded, combined_kernel)


class JBULearnedRangeNoSoftmax(torch.nn.Module):

    def __init__(self, guidance_dim, feat_dim, key_dim, scale=2, radius=3, norm="euclidean"):
        super().__init__()
        self.scale = scale
        self.radius = radius
        self.diameter = self.radius * 2 + 1

        self.guidance_dim = guidance_dim
        self.key_dim = key_dim
        self.feat_dim = feat_dim
        self.norm = norm

        self.range_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim, key_dim, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(key_dim, key_dim, 1, 1),
        )

        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim + self.diameter ** 2, self.diameter ** 2, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(self.diameter ** 2, self.diameter ** 2, 1, 1),
        )
        self.sigma_range = nn.Parameter(torch.tensor(1.0))

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))

    def get_range_kernel(self, x):
        GB, GC, GH, GW = x.shape
        proj_x = self.range_proj(x)
        proj_x_padded = F.pad(proj_x, pad=[self.radius] * 4, mode='reflect')
        queries = torch.nn.Unfold(self.diameter)(proj_x_padded) \
            .reshape((GB, self.key_dim, self.diameter * self.diameter, GH, GW)) \
            .permute(0, 1, 3, 4, 2)

        if self.norm == "euclidean":
            dists = (queries - proj_x.unsqueeze(-1)).square().sum(1).permute(0, 3, 1, 2)
        elif self.norm == "cosine":
            if GC == 1:
                range_kernel = 1 - torch.einsum("bchwp,bchw->bphw", queries, proj_x)
                range_kernel -= torch.amin(range_kernel, dim=(1, 2, 3), keepdim=True)
                range_kernel /= torch.amax(range_kernel, dim=(1, 2, 3), keepdim=True)
            else:
                range_kernel = 2 - torch.einsum("bchwp,bchw->bphw", queries, proj_x)
            dists = range_kernel ** 2
        else:
            raise ValueError(f"Unknown norm {self.norm}")

        return torch.exp(- dists / (2 * self.sigma_range ** 2))

    def get_spatial_kernel(self):
        dist_range = torch.linspace(-1, 1, self.diameter, device=self.sigma_spatial.device)
        x, y = torch.meshgrid(dist_range, dist_range)
        patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        return torch.exp(- patch.square().sum(0) / (2 * self.sigma_spatial ** 2)) \
            .reshape(1, self.diameter * self.diameter, 1, 1)

    def forward(self, source, guidance):
        GB, GC, GH, GW = guidance.shape
        SB, SC, SH, SQ = source.shape
        assert (SB == GB)

        spatial_kernel = self.get_spatial_kernel()
        range_kernel = self.get_range_kernel(guidance)

        combined_kernel = range_kernel * spatial_kernel
        combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(1e-7)

        combined_kernel += .1 * self.fixup_proj(torch.cat([combined_kernel, guidance], dim=1))
        combined_kernel = combined_kernel.permute(0, 2, 3, 1) \
            .reshape(GB, GH, GW, self.diameter, self.diameter)

        hr_source = torch.nn.Upsample((GH, GW), mode='bicubic', align_corners=False)(source)
        hr_source_padded = F.pad(hr_source, pad=[self.radius] * 4, mode='reflect')

        # (B C, H+Pad, W+Pad) x (B, H, W, KH, KW) -> BCHW
        return AdaptiveConv.apply(hr_source_padded, combined_kernel)


class JBULearnedRange2(torch.nn.Module):

    def __init__(self, guidance_dim, feat_dim, key_dim, scale=2, radius=3):
        super().__init__()
        self.scale = scale
        self.radius = radius
        self.diameter = self.radius * 2 + 1

        self.guidance_dim = guidance_dim
        self.key_dim = key_dim
        self.feat_dim = feat_dim

        self.range_temp = nn.Parameter(torch.tensor(1.0))
        self.range_proj = torch.nn.Sequential(
            torch.nn.Conv2d(self.guidance_dim + 4, key_dim, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(key_dim, key_dim, 1, 1),
        )

        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Conv2d(self.guidance_dim + self.diameter ** 2, self.diameter ** 2, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(self.diameter ** 2, self.diameter ** 2, 1, 1),
        )

        self.source_proj = torch.nn.Sequential(
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(feat_dim, 4 * 2 * 2, 1),
            torch.nn.PixelShuffle(2)
        )

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))

    def get_range_kernel(self, guidance):
        GB, GC, GH, GW = guidance.shape
        proj_guidance = self.range_proj(guidance)
        proj_guidance_padded = F.pad(proj_guidance, pad=[self.radius] * 4, mode='reflect')
        queries = torch.nn.Unfold(self.diameter)(proj_guidance_padded) \
            .reshape((GB, self.key_dim, self.diameter * self.diameter, GH, GW)) \
            .permute(0, 1, 3, 4, 2)
        pos_temp = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
        return F.softmax(pos_temp * torch.einsum("bchwp,bchw->bphw", queries, proj_guidance), dim=1)

    def get_spatial_kernel(self):
        dist_range = torch.linspace(-1, 1, self.diameter, device=self.sigma_spatial.device)
        x, y = torch.meshgrid(dist_range, dist_range)
        patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        return torch.exp(- patch.square().sum(0) / (2 * self.sigma_spatial ** 2)) \
            .reshape(1, self.diameter * self.diameter, 1, 1)

    def forward(self, source, guidance):
        GB, GC, GH, GW = guidance.shape
        SB, SC, SH, SQ = source.shape
        assert (SB == GB)

        aug_guidance = torch.cat([guidance, self.source_proj(source) * .2], dim=1)

        spatial_kernel = self.get_spatial_kernel()
        range_kernel = self.get_range_kernel(aug_guidance)

        combined_kernel = range_kernel * spatial_kernel
        combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(1e-7)

        combined_kernel += .1 * self.fixup_proj(torch.cat([combined_kernel, guidance], dim=1))
        combined_kernel = combined_kernel.permute(0, 2, 3, 1) \
            .reshape(GB, GH, GW, self.diameter, self.diameter)

        hr_source = torch.nn.Upsample((GH, GW), mode='bicubic', align_corners=False)(source)
        hr_source_padded = F.pad(hr_source, pad=[self.radius] * 4, mode='reflect')

        # (B C, H+Pad, W+Pad) x (B, H, W, KH, KW) -> BCHW
        return AdaptiveConv.apply(hr_source_padded, combined_kernel)


class JBUAdapter(torch.nn.Module):

    def __init__(self, model, jbu=JBU()):
        super().__init__()
        self.model = model
        self.jbu = jbu

    def forward(self, img):
        feats = self.model(img)
        return self.jbu(img, source=feats)


class RConvModel:
    pass


def nearest_interpolate(x, scale_factor=2):
    b, h, w, c = x.shape  # channels last
    return x.repeat(1, 1, 1, scale_factor ** 2).reshape(b, h, w, scale_factor, scale_factor, c).permute(
        0, 1, 3, 2, 4, 5).reshape(b, scale_factor * h, scale_factor * w, c)


class SAPAExpModule(nn.Module):
    def __init__(self, dim_y, dim_x=None, out_dim=None,
                 q_mode='encoder_only', v_embed=False,
                 up_factor=2, up_kernel_size=5, embedding_dim=64,
                 qkv_bias=True, norm=nn.LayerNorm):
        super().__init__()
        dim_x = dim_x if dim_x is not None else dim_y
        out_dim = out_dim if out_dim is not None else dim_x

        self.up_factor = up_factor
        self.up_kernel_size = up_kernel_size
        self.embedding_dim = embedding_dim

        self.norm_y = norm(dim_y)
        self.norm_x = norm(dim_x)

        self.q_mode = q_mode
        if q_mode == 'encoder_only':
            self.q = nn.Linear(dim_y, embedding_dim, bias=qkv_bias)
        elif q_mode == 'cat':
            self.q = nn.Linear(dim_x + dim_y, embedding_dim, bias=qkv_bias)
        elif q_mode == 'gate':
            self.qy = nn.Linear(dim_y, embedding_dim, bias=qkv_bias)
            self.qx = nn.Linear(dim_x, embedding_dim, bias=qkv_bias)
            self.gate = nn.Linear(dim_x, 1, bias=qkv_bias)
        else:
            raise NotImplementedError

        self.k = nn.Linear(dim_x, embedding_dim, bias=qkv_bias)

        if v_embed or out_dim != dim_x:
            self.v = nn.Linear(dim_x, out_dim, bias=qkv_bias)

        self.apply(self._init_weights)

    def forward(self, y, x):
        y = y.permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 2, 3, 1).contiguous()
        y = self.norm_y(y)
        x_ = self.norm_x(x)

        if self.q_mode == 'encoder_only':
            q = self.q(y)
        elif self.q_mode == 'cat':
            q = self.q(torch.cat([y, nearest_interpolate(x, self.up_factor)], dim=-1))
        elif self.q_mode == 'gate':
            gate = nearest_interpolate(torch.sigmoid(self.gate(x_)), self.up_factor)
            q = gate * self.qy(y) + (1 - gate) * self.qx(nearest_interpolate(x, self.up_factor))
        else:
            raise NotImplementedError

        k = self.k(x_)

        if hasattr(self, 'v'):
            x = self.v(x_)

        return self.attention(q, k, x).permute(0, 3, 1, 2).contiguous()

    def attention(self, q, k, v):
        from sapa import sim, atn

        attn = F.softmax(sim(q, k, self.up_kernel_size, self.up_factor), dim=-1)
        return atn(attn, v, self.up_kernel_size, self.up_factor)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class SAPAModule(nn.Module):
    def __init__(self, dim_y, dim_x=None,
                 up_factor=2, up_kernel_size=5, embedding_dim=64,
                 qkv_bias=True, norm=nn.LayerNorm):
        super().__init__()
        dim_x = dim_x if dim_x is not None else dim_y

        self.up_factor = up_factor
        self.up_kernel_size = up_kernel_size
        self.embedding_dim = embedding_dim

        self.norm_y = norm(dim_y)
        self.norm_x = norm(dim_x)

        self.q = nn.Linear(dim_y, embedding_dim, bias=qkv_bias)
        self.k = nn.Linear(dim_x, embedding_dim, bias=qkv_bias)

        self.apply(self._init_weights)

    def forward(self, y, x):
        y = y.permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 2, 3, 1).contiguous()
        y = self.norm_y(y)
        x_ = self.norm_x(x)

        q = self.q(y)
        k = self.k(x_)

        return self.attention(q, k, x).permute(0, 3, 1, 2).contiguous()

    def attention(self, q, k, v):
        from sapa import sim, atn

        attn = F.softmax(sim(q, k, self.up_kernel_size, self.up_factor), dim=-1)
        return atn(attn, v, self.up_kernel_size, self.up_factor)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class CarafeGuidedUpsampler(torch.nn.Module):

    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.up1 = CARAFEPack(dim + 3, up_kernel=3, up_group=1, scale_factor=2)
        self.up2 = CARAFEPack(dim + 3, up_kernel=3, up_group=1, scale_factor=2)
        self.up3 = CARAFEPack(dim + 3, up_kernel=3, up_group=1, scale_factor=2)
        self.up4 = CARAFEPack(dim + 3, up_kernel=3, up_group=1, scale_factor=2)
        self.proj = torch.nn.Conv2d(dim + 3, dim, 1, 1)

    def apply_conv(self, source, guidance, up):
        _, _, h, w = source.shape
        small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        return up(torch.cat([source, small_guidance], dim=1))[:, :self.dim, :, :]

    def forward(self, source, guidance):
        source_2 = self.apply_conv(source, guidance, self.up1)
        source_4 = self.apply_conv(source_2, guidance, self.up2)
        source_8 = self.apply_conv(source_4, guidance, self.up3)
        source_16 = self.apply_conv(source_8, guidance, self.up4)
        return self.proj(torch.cat([source_16, guidance], dim=1))


class CarafeUpsampler(torch.nn.Module):

    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up1 = CARAFEPack(dim, up_kernel=3, up_group=1, scale_factor=2)
        self.up2 = CARAFEPack(dim, up_kernel=3, up_group=1, scale_factor=2)
        self.up3 = CARAFEPack(dim, up_kernel=3, up_group=1, scale_factor=2)
        self.up4 = CARAFEPack(dim, up_kernel=3, up_group=1, scale_factor=2)

    def forward(self, source, guidance):
        source_2 = self.up1(source)
        source_4 = self.up2(source_2)
        source_8 = self.up3(source_4)
        source_16 = self.up4(source_8)
        return source_16


class SAPAUpsampler(torch.nn.Module):
    def __init__(self, dim_x, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up1 = SAPAModule(dim_x=dim_x, dim_y=3)
        self.up2 = SAPAModule(dim_x=dim_x, dim_y=3)
        self.up3 = SAPAModule(dim_x=dim_x, dim_y=3)
        self.up4 = SAPAModule(dim_x=dim_x, dim_y=3)

    def adapt_guidance(self, source, guidance):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        return small_guidance

    def forward(self, source, guidance):
        source_2 = self.up1(self.adapt_guidance(source, guidance), source)
        source_4 = self.up2(self.adapt_guidance(source_2, guidance), source_2)
        source_8 = self.up3(self.adapt_guidance(source_4, guidance), source_4)
        source_16 = self.up4(self.adapt_guidance(source_8, guidance), source_8)
        return source_16


class LayeredResizeConv(torch.nn.Module):

    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv3 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv4 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")

    def apply_conv(self, source, guidance, conv, activation):
        big_source = F.interpolate(source, scale_factor=2, mode="bilinear")
        _, _, h, w = big_source.shape
        small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        output = activation(conv(torch.cat([big_source, small_guidance], dim=1)))
        return big_source + output

    def forward(self, source, guidance):
        source_2 = self.apply_conv(source, guidance, self.conv1, F.relu)
        source_4 = self.apply_conv(source_2, guidance, self.conv2, F.relu)
        source_8 = self.apply_conv(source_4, guidance, self.conv3, F.relu)
        source_16 = self.apply_conv(source_8, guidance, self.conv4, lambda x: x)
        return source_16


class LayeredResizeConvResidual(torch.nn.Module):

    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv3 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv4 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")

        self.rconv1 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.rconv2 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.rconv3 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.rconv4 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")

    def apply_conv(self, source, guidance, conv, activation):
        big_source = F.interpolate(source, scale_factor=2, mode="bilinear")
        _, _, h, w = big_source.shape
        small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        output = activation(conv(torch.cat([big_source, small_guidance], dim=1)))
        return big_source + output

    def apply_rconv(self, source, guidance, conv, activation):
        _, _, h, w = source.shape
        small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        output = activation(conv(torch.cat([source, small_guidance], dim=1)))
        return source + output

    def forward(self, source, guidance):
        source_2 = self.apply_conv(source, guidance, self.conv1, F.relu)
        source_2_2 = self.apply_rconv(source_2, guidance, self.rconv1, F.relu)
        source_4 = self.apply_conv(source_2_2, guidance, self.conv2, F.relu)
        source_4_2 = self.apply_rconv(source_4, guidance, self.rconv2, F.relu)
        source_8 = self.apply_conv(source_4_2, guidance, self.conv3, F.relu)
        source_8_2 = self.apply_rconv(source_8, guidance, self.rconv3, F.relu)
        source_16 = self.apply_conv(source_8_2, guidance, self.conv4, lambda x: x)
        source_16_2 = self.apply_rconv(source_16, guidance, self.rconv4, F.relu)
        return source_16_2


class LayeredResizeConvWithErrors(torch.nn.Module):

    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(dim + 6, dim, kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(dim + 6, dim, kernel_size, padding="same")
        self.conv3 = torch.nn.Conv2d(dim + 6, dim, kernel_size, padding="same")
        self.conv4 = torch.nn.Conv2d(dim + 6, dim, kernel_size, padding="same")

    def apply_conv(self, source, guidance, conv, activation):
        big_source = F.interpolate(source, scale_factor=2, mode="bilinear")
        _, _, h, w = big_source.shape
        _, _, sh, sw = source.shape

        small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        error_guidance = small_guidance - F.interpolate(
            F.interpolate(guidance, (sh, sw), mode="bilinear"),
            scale_factor=2, mode="bilinear")
        error_guidance -= error_guidance.mean(dim=[2, 3], keepdim=True)
        error_guidance /= error_guidance.std(dim=[2, 3], keepdim=True).clamp_min(0.001)

        output = activation(conv(torch.cat([big_source, small_guidance, error_guidance], dim=1)))
        return big_source + output

    def forward(self, source, guidance):
        source_2 = self.apply_conv(source, guidance, self.conv1, F.relu)
        source_4 = self.apply_conv(source_2, guidance, self.conv2, F.relu)
        source_8 = self.apply_conv(source_4, guidance, self.conv3, F.relu)
        source_16 = self.apply_conv(source_8, guidance, self.conv4, lambda x: x)
        return source_16


class LayeredDeConv(torch.nn.Module):

    def get_deconv(self, dim):
        return torch.nn.ConvTranspose2d(dim + 3, dim, 2, stride=2)

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = self.get_deconv(dim)
        self.conv2 = self.get_deconv(dim)
        self.conv3 = self.get_deconv(dim)
        self.conv4 = self.get_deconv(dim)

    def apply_conv(self, source, guidance, conv, activation):
        _, _, h, w = source.shape
        small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        output = activation(conv(torch.cat([source, small_guidance], dim=1)))
        return output

    def forward(self, source, guidance):
        source_2 = self.apply_conv(source, guidance, self.conv1, lambda x: x)
        source_4 = self.apply_conv(source_2, guidance, self.conv2, lambda x: x)
        source_8 = self.apply_conv(source_4, guidance, self.conv3, lambda x: x)
        source_16 = self.apply_conv(source_8, guidance, self.conv4, lambda x: x)
        return source_16


class LayeredResizeConvWithImplicits(torch.nn.Module):

    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_dim = dim + 100 + 3
        self.conv1 = torch.nn.Conv2d(in_dim, dim, kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(in_dim, dim, kernel_size, padding="same")
        self.conv3 = torch.nn.Conv2d(in_dim, dim, kernel_size, padding="same")
        self.conv4 = torch.nn.Conv2d(in_dim, dim, kernel_size, padding="same")
        self.implicit_featurizer = ImplicitFeaturizer()

    def apply_conv(self, source, guidance, conv, activation, residual):
        big_source = F.interpolate(source, scale_factor=2, mode="bilinear")
        _, _, h, w = big_source.shape
        small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        guidance_feats = self.implicit_featurizer(small_guidance)

        output = activation(conv(torch.cat([big_source, guidance_feats], dim=1)))

        if residual:
            return big_source + output
        else:
            return output

    def forward(self, source, guidance):
        source_2 = self.apply_conv(source, guidance, self.conv1, F.relu, True)
        source_4 = self.apply_conv(source_2, guidance, self.conv2, F.relu, True)
        source_8 = self.apply_conv(source_4, guidance, self.conv3, F.relu, True)
        source_16 = self.apply_conv(source_8, guidance, self.conv4, lambda x: x, False)
        return source_16


class FullImplicit(torch.nn.Module):

    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_dim = 200 + 3
        self.conv1 = torch.nn.Conv2d(in_dim, in_dim, kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(in_dim, in_dim, kernel_size, padding="same")
        self.conv3 = torch.nn.Conv2d(in_dim, in_dim, kernel_size, padding="same")
        self.conv4 = torch.nn.Conv2d(in_dim, dim, kernel_size, padding="same")
        self.implicit_featurizer = ImplicitFeaturizer(n_freqs=20)

    def apply_conv(self, feats, conv, activation, residual):
        output = activation(conv(feats))
        if residual:
            return feats + output
        else:
            return output

    def forward(self, source, guidance):
        feats1 = self.implicit_featurizer(guidance)
        feats2 = self.apply_conv(feats1, self.conv1, F.relu, True)
        feats3 = self.apply_conv(feats2, self.conv2, F.relu, True)
        feats4 = self.apply_conv(feats3, self.conv3, F.relu, True)
        feats5 = self.apply_conv(feats4, self.conv4, lambda x: x, False)
        return feats5


class DeConv(torch.nn.Module):

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 16, stride=16)

    def forward(self, source, guidance):
        return self.conv(source)


class LayeredResizeConv2(torch.nn.Module):

    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv3 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv4 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv5 = torch.nn.Conv2d(dim + 3, dim, 1, padding="same")

        self.jbu1 = get_jbu()
        self.jbu2 = get_jbu()
        self.jbu3 = get_jbu()
        self.jbu4 = get_jbu()

    def apply_conv(self, source, guidance, conv, activation, jbu, residual):
        if jbu is not None:
            big_source = jbu(source, guidance)
            _, _, h, w = big_source.shape
            small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        else:
            big_source = source
            small_guidance = guidance

        output = activation(conv(torch.cat([big_source, small_guidance], dim=1)))

        if residual:
            return big_source + output
        else:
            return output

    def forward(self, source, guidance):
        source_2 = self.apply_conv(source, guidance, self.conv1, F.relu, self.jbu1, True)
        source_4 = self.apply_conv(source_2, guidance, self.conv2, F.relu, self.jbu2, True)
        source_8 = self.apply_conv(source_4, guidance, self.conv3, F.relu, self.jbu3, True)
        source_16 = self.apply_conv(source_8, guidance, self.conv4, F.relu, self.jbu4, True)
        source_16 = self.apply_conv(source_16, guidance, self.conv5, lambda x: x, None, False)

        return source_16


class JBUStack(torch.nn.Module):

    def __init__(self, feat_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up1 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up2 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up3 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up4 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=1))

    def upsample(self, source, guidance, up):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        upsampled = up(source, small_guidance)
        return upsampled

    def forward(self, source, guidance):
        source_2 = self.upsample(source, guidance, self.up1)
        source_4 = self.upsample(source_2, guidance, self.up2)
        source_8 = self.upsample(source_4, guidance, self.up3)
        source_16 = self.upsample(source_8, guidance, self.up4)
        return self.fixup_proj(source_16) * 0.1 + source_16


def get_jbu():
    return JBU(
        scale=2,
        radius=3,
        sigma_spatial=1.0,
        sigma_range=.3,
        epsilon=1e-8,
        no_grad=False)


class LearnedJBUBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.jbu = get_jbu()
        self.guidance_conv = torch.nn.Conv2d(dim + 3, 30, 1, padding="same")
        self.final_conv = id_conv(dim)

    def forward(self, source, guidance):
        big_source = F.interpolate(source, scale_factor=2, mode="bilinear")
        _, _, h, w = big_source.shape
        small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        new_guidance = self.guidance_conv(torch.cat([small_guidance, big_source], dim=1))
        return self.final_conv(self.jbu(source, new_guidance))
        # return self.jbu(source, new_guidance)


class LearnedLayeredJBU(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.jbu1 = LearnedJBUBlock(dim)
        self.jbu2 = LearnedJBUBlock(dim)
        self.jbu3 = LearnedJBUBlock(dim)
        self.jbu4 = LearnedJBUBlock(dim)

    def forward(self, source, guidance):
        source_2 = self.jbu1(source, guidance)
        source_4 = self.jbu2(source_2, guidance)
        source_8 = self.jbu3(source_4, guidance)
        source_16 = self.jbu4(source_8, guidance)
        return source_16


class LayeredJBU(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.jbu1 = get_jbu()
        self.jbu2 = get_jbu()
        self.jbu3 = get_jbu()
        self.jbu4 = get_jbu()

    def forward(self, source, guidance):
        source_2 = self.jbu1(source, guidance)
        source_4 = self.jbu2(source_2, guidance)
        source_8 = self.jbu3(source_4, guidance)
        source_16 = self.jbu4(source_8, guidance)
        return source_16

    def add_logs(self, log_func):
        log_func("layered_jbu/1/range", self.jbu1.sigma_range)
        log_func("layered_jbu/1/spatial", self.jbu1.sigma_spatial)
        log_func("layered_jbu/2/range", self.jbu2.sigma_range)
        log_func("layered_jbu/2/spatial", self.jbu2.sigma_spatial)
        log_func("layered_jbu/3/range", self.jbu3.sigma_range)
        log_func("layered_jbu/3/spatial", self.jbu3.sigma_spatial)
        log_func("layered_jbu/4/range", self.jbu4.sigma_range)
        log_func("layered_jbu/4/spatial", self.jbu4.sigma_spatial)


class IteratedJBU(torch.nn.Module):

    def __init__(self, n=5):
        super().__init__()

        self.n = n
        self.jbu1 = JBU(
            scale=1,
            radius=5,
            sigma_spatial=3.0,
            sigma_range=.2,
            epsilon=1e-8)

    def forward(self, source, guidance):
        source = F.interpolate(source, scale_factor=16, mode="bilinear")
        for i in range(self.n):
            source = self.jbu1(source, guidance)

        return source


class Bilinear(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, feats, img):
        _, _, h, w = img.shape
        return F.interpolate(feats, (h, w), mode="bilinear")


class ResizeConv(torch.nn.Module):

    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")

    def forward(self, feats, img):
        _, _, h, w = img.shape
        big_feats = F.interpolate(feats, (h, w), mode="bilinear")
        return self.conv(torch.cat([big_feats, img], dim=1)) + big_feats


class LearnedLayeredResizeConv2(torch.nn.Module):

    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv3 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv4 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv5 = torch.nn.Conv2d(dim + 3, dim, 1, padding="same")

        self.jbu1 = LearnedJBUBlock(dim)
        self.jbu2 = LearnedJBUBlock(dim)
        self.jbu3 = LearnedJBUBlock(dim)
        self.jbu4 = LearnedJBUBlock(dim)

    def apply_conv(self, source, guidance, conv, activation, jbu, residual):
        if jbu is not None:
            big_source = jbu(source, guidance)
            _, _, h, w = big_source.shape
            small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        else:
            big_source = source
            small_guidance = guidance

        output = activation(conv(torch.cat([big_source, small_guidance], dim=1)))

        if residual:
            return big_source + output
        else:
            return output

    def forward(self, source, guidance):
        source_2 = self.apply_conv(source, guidance, self.conv1, F.relu, self.jbu1, True)
        source_4 = self.apply_conv(source_2, guidance, self.conv2, F.relu, self.jbu2, True)
        source_8 = self.apply_conv(source_4, guidance, self.conv3, F.relu, self.jbu3, True)
        source_16 = self.apply_conv(source_8, guidance, self.conv4, F.relu, self.jbu4, True)
        source_16 = self.apply_conv(source_16, guidance, self.conv5, lambda x: x, None, False)

        return source_16


def get_upsampler(upsampler, dim):
    if upsampler == 'layered_resize_conv':
        return LayeredResizeConv(dim, 1)
    elif upsampler == 'layered_deconv':
        return LayeredDeConv(dim)
    elif upsampler == 'deconv':
        return DeConv(dim)
    elif upsampler == 'bilinear':
        return Bilinear()
    elif upsampler == 'iterated_jbu':
        return IteratedJBU()
    elif upsampler == 'learned_layered_jbu':
        return LearnedLayeredJBU(dim)
    elif upsampler == 'layered_jbu':
        return LayeredJBU()
    elif upsampler == 'jbu_stack':
        return JBUStack(dim)
    elif upsampler == 'layered_resize_conv_2':
        return LayeredResizeConv2(dim, 1)
    elif upsampler == 'learned_layered_resize_conv_2':
        return LearnedLayeredResizeConv2(dim, 1)
    elif upsampler == 'layered_resize_conv_with_implicits':
        return LayeredResizeConvWithImplicits(dim, 1)
    elif upsampler == 'full_implicit':
        return FullImplicit(dim, 1)
    elif upsampler == 'carafe':
        return CarafeUpsampler(dim, 1)
    elif upsampler == 'guided_carafe':
        return CarafeGuidedUpsampler(dim, 1)
    elif upsampler == 'sapa':
        return SAPAUpsampler(dim_x=dim)
    else:
        raise ValueError(f"Unknown upsampler {upsampler}")
