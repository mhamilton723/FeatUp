import torch


def entropy(t):
    return -(t * torch.log(t.clamp_min(.0000001))).sum(dim=[-1, -2, -3]).mean()


def total_variation(img):
    b, c, h, w = img.size()
    return ((img[:, :, 1:, :] - img[:, :, :-1, :]).square().sum() +
            (img[:, :, :, 1:] - img[:, :, :, :-1]).square().sum()) / (b * c * h * w)


class SampledCRFLoss(torch.nn.Module):

    def __init__(self, n_samples, alpha, beta, gamma, w1, w2, shift):
        super(SampledCRFLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w1 = w1
        self.w2 = w2
        self.n_samples = n_samples
        self.shift = shift

    def forward(self, guidance, features):
        device = features.device
        assert (guidance.shape[0] == features.shape[0])
        assert (guidance.shape[2:] == features.shape[2:])
        h = guidance.shape[2]
        w = guidance.shape[3]

        coords = torch.cat([
            torch.randint(0, h, size=[1, self.n_samples], device=device),
            torch.randint(0, w, size=[1, self.n_samples], device=device)], 0)
        norm_coords = coords / torch.tensor([h, w], device=guidance.device).unsqueeze(-1)

        selected_guidance = guidance[:, :, coords[0, :], coords[1, :]]

        coord_diff = (norm_coords.unsqueeze(-1) - norm_coords.unsqueeze(-2)).square().sum(0).unsqueeze(0)
        guidance_diff = (selected_guidance.unsqueeze(-1) - selected_guidance.unsqueeze(-2)).square().sum(1)

        sim_kernel = self.w1 * torch.exp(- coord_diff / (2 * self.alpha) - guidance_diff / (2 * self.beta)) + \
                     self.w2 * torch.exp(- coord_diff / (2 * self.gamma)) - self.shift

        # selected_clusters = F.normalize(features[:, :, coords[0, :], coords[1, :]], dim=1)
        # cluster_sims = torch.einsum("bcn,bcm->bnm", selected_clusters, selected_clusters)
        selected_feats = features[:, :, coords[0, :], coords[1, :]]
        feat_diff = (selected_feats.unsqueeze(-1) - selected_feats.unsqueeze(-2)).square().sum(1)

        return (feat_diff * sim_kernel).mean()


class TVLoss(torch.nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, img):
        b, c, h, w = img.size()
        return ((img[:, :, 1:, :] - img[:, :, :-1, :]).square().sum() +
                (img[:, :, :, 1:] - img[:, :, :, :-1]).square().sum()) / (b * c * h * w)
