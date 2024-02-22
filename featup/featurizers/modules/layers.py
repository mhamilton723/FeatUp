import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

__all__ = ['forward_hook', 'AdaptiveAvgPool2d', 'Add', 'AvgPool2d', 'BatchNorm2d', 'Clone', 'Conv2d', 'ConvTranspose2d',
           'Dropout', 'Identity', 'LeakyReLU', 'Linear', 'MaxPool2d', 'Multiply', 'ReLU', 'Sequential', 'safe_divide',
           'ZeroPad2d', 'LayerNorm', 'GELU', 'einsum', 'Softmax']


def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha=1):
        return R


class RelPropSimple(RelProp):
    def relprop(self, R, alpha=1):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * C[0]
        return outputs


class Identity(nn.Identity, RelProp):
    pass


class ReLU(nn.ReLU, RelProp):
    pass


class GELU(nn.GELU, RelProp):
    pass

class LeakyReLU(nn.LeakyReLU, RelProp):
    pass

class Softmax(nn.Softmax, RelProp):
    pass

class einsum(RelPropSimple):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation
    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)

class Dropout(nn.Dropout, RelProp):
    pass


class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass

class LayerNorm(nn.LayerNorm, RelProp):
    pass

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelProp):
    def relprop(self, R, alpha=1):
        px = torch.clamp(self.X, min=0)

        def f(x1):
            Z1 = F.adaptive_avg_pool2d(x1, self.output_size)
            S1 = safe_divide(R, Z1)
            C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            return C1

        activator_relevances = f(px)
        out = activator_relevances
        return out


class ZeroPad2d(nn.ZeroPad2d, RelPropSimple):
    def relprop(self, R, alpha=1):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        outputs = self.X * C[0]
        return outputs


class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs


class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha = 1):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C

        return R


class Multiply(RelPropSimple):
    def forward(self, inputs):
        return torch.mul(*inputs)

    def relprop(self, R, alpha=1):
        x0 = torch.clamp(self.X[0], min=0)
        x1 = torch.clamp(self.X[1], min=0)
        x = [x0, x1]
        Z = self.forward(x)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, x, S)
        outputs = []
        outputs.append(x[0] * C[0])
        outputs.append(x[1] * C[1])
        return outputs

class Sequential(nn.Sequential):
    def relprop(self, R, alpha=1):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R



class BatchNorm2d(nn.BatchNorm2d, RelProp):
    def relprop(self, R, alpha=1):
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R


class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha=1):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        # def f(w1, w2, x1, x2):
        #     Z1 = F.linear(x1, w1)
        #     Z2 = F.linear(x2, w2)
        #     S1 = safe_divide(R, Z1)
        #     S2 = safe_divide(R, Z2)
        #     C1 = x1 * self.gradprop(Z1, x1, S1)[0]
        #     C2 = x2 * self.gradprop(Z2, x2, S2)[0]
        #     return C1 #+ C2

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            Z = Z1 + Z2
            S = safe_divide(R, Z)
            C1 = x1 * self.gradprop(Z1, x1, S)[0]
            C2 = x2 * self.gradprop(Z2, x2, S)[0]
            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        out = alpha * activator_relevances - beta * inhibitor_relevances

        return out



class Conv2d(nn.Conv2d, RelProp):

    def relprop(self, R, alpha=1):
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + \
                torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.X * 0 + \
                torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
                Z2 = F.conv2d(x2, w2, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
                Z = Z1 + Z2
                S = safe_divide(R, Z)
                C1 = x1 * self.gradprop(Z1, x1, S)[0]
                C2 = x2 * self.gradprop(Z2, x2, S)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R



class ConvTranspose2d(nn.ConvTranspose2d, RelProp):
    def relprop(self, R, alpha=1):
        pw = torch.clamp(self.weight, min=0)
        px = torch.clamp(self.X, min=0)

        def f(w1, x1):
            Z1 = F.conv_transpose2d(x1, w1, bias=None, stride=self.stride, padding=self.padding,
                                    output_padding=self.output_padding)
            S1 = safe_divide(R, Z1)
            C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            return C1

        activator_relevances = f(pw, px)
        R = activator_relevances
        return R



if __name__ == '__main__':
    convt = ConvTranspose2d(100, 50, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False).cuda()

    rand = torch.rand((1, 100, 224, 224)).cuda()
    out = convt(rand)
    rel = convt.relprop(out)

    print(out.shape)
