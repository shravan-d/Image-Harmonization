import torch
import torch.nn.functional as func
from torch import nn
import functools


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.return_mask = True
        super(PartialConv2d, self).__init__(*args, **kwargs)
        self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] \
                             * self.weight_maskUpdater.shape[3]
        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, x, mask_in=None):
        assert len(x.shape) == 4
        if mask_in is not None or self.last_size != tuple(x.shape):
            self.last_size = tuple(x.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)

                if mask_in is None:
                    mask = torch.ones(1, 1, x.data.shape[2], x.data.shape[3]).to(x)
                else:
                    mask = mask_in

                self.update_mask = func.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                               padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(x, mask) if mask_in is not None else x)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class StyleEncoder(nn.Module):
    def __init__(self, style_dim, norm_layer=nn.BatchNorm2d):
        super(StyleEncoder, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        ndf = 64
        n_layers = 6
        kw = 3
        padw = 0
        self.conv1f = PartialConv2d(3, ndf, kernel_size=kw, stride=2, padding=padw)
        self.relu1 = nn.ReLU(True)
        nf_mult = 1
        nf_mult_prev = 1

        n = 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv2f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                                    bias=use_bias)
        self.norm2f = norm_layer(ndf * nf_mult)
        self.relu2 = nn.ReLU(True)

        n = 2
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv3f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                                    bias=use_bias)
        self.norm3f = norm_layer(ndf * nf_mult)
        self.relu3 = nn.ReLU(True)

        n = 3
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv4f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                                    bias=use_bias)
        self.norm4f = norm_layer(ndf * nf_mult)
        self.relu4 = nn.ReLU(True)

        n = 4
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv5f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                                    bias=use_bias)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.convs = nn.Conv2d(ndf * nf_mult, style_dim, kernel_size=1, stride=1)

    def forward(self, input, mask):
        """Standard forward."""
        xb = input
        mb = mask

        xb, mb = self.conv1f(xb, mb)
        xb = self.relu1(xb)
        xb, mb = self.conv2f(xb, mb)
        xb = self.norm2f(xb)
        xb = self.relu2(xb)
        xb, mb = self.conv3f(xb, mb)
        xb = self.norm3f(xb)
        xb = self.relu3(xb)
        xb, mb = self.conv4f(xb, mb)
        xb = self.norm4f(xb)
        xb = self.relu4(xb)
        xb, mb = self.conv5f(xb, mb)
        xb = self.avg_pooling(xb)
        s = self.convs(xb)
        return s
