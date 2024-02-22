import torch
import torch.nn as nn


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


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
