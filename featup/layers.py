import torch


def id_conv(dim, strength=.9):
    conv = torch.nn.Conv2d(dim, dim, 1, padding="same")
    start_w = conv.weight.data
    conv.weight.data = torch.nn.Parameter(
        torch.eye(dim, device=start_w.device).unsqueeze(-1).unsqueeze(-1) * strength + start_w * (1 - strength))
    conv.bias.data = torch.nn.Parameter(conv.bias.data * (1 - strength))
    return conv


class ImplicitFeaturizer(torch.nn.Module):

    def __init__(self, color_feats=True, n_freqs=10, learn_bias=False, time_feats=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_feats = color_feats
        self.time_feats = time_feats
        self.n_freqs = n_freqs
        self.learn_bias = learn_bias

        self.dim_multiplier = 2

        if self.color_feats:
            self.dim_multiplier += 3

        if self.time_feats:
            self.dim_multiplier += 1

        if self.learn_bias:
            self.biases = torch.nn.Parameter(torch.randn(2, self.dim_multiplier, n_freqs).to(torch.float32))

    def forward(self, original_image):
        b, c, h, w = original_image.shape
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.cat([t.unsqueeze(0) for t in torch.meshgrid([grid_h, grid_w])]).unsqueeze(0)
        feats = torch.broadcast_to(feats, (b, feats.shape[1], h, w))

        if self.color_feats:
            feat_list = [feats, original_image]
        else:
            feat_list = [feats]

        feats = torch.cat(feat_list, dim=1).unsqueeze(1)
        freqs = torch.exp(torch.linspace(-2, 10, self.n_freqs, device=original_image.device)) \
            .reshape(1, self.n_freqs, 1, 1, 1)
        feats = (feats * freqs)

        if self.learn_bias:
            sin_feats = feats + self.biases[0].reshape(1, self.n_freqs, self.dim_multiplier, 1, 1)
            cos_feats = feats + self.biases[1].reshape(1, self.n_freqs, self.dim_multiplier, 1, 1)
        else:
            sin_feats = feats
            cos_feats = feats

        sin_feats = sin_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)
        cos_feats = cos_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)

        if self.color_feats:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats), original_image]
        else:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats)]

        return torch.cat(all_feats, dim=1)


class MinMaxScaler(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        c = x.shape[1]
        flat_x = x.permute(1, 0, 2, 3).reshape(c, -1)
        flat_x_min = flat_x.min(dim=-1).values.reshape(1, c, 1, 1)
        flat_x_scale = flat_x.max(dim=-1).values.reshape(1, c, 1, 1) - flat_x_min
        return ((x - flat_x_min) / flat_x_scale.clamp_min(0.0001)) - .5


class ChannelNorm(torch.nn.Module):

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        new_x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return new_x
