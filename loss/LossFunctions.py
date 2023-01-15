import torch.nn as nn
import torch
from torchvision.transforms import functional
from torchvision.ops import masks_to_boxes
import math

mse_criterion = torch.nn.MSELoss(reduction='mean')
eps = 1e-06


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.loss = 0

    def forward(self, features, targets, weights=None):
        if weights is None:
            weights = [1 / len(features)] * len(features)

        for f, t, w in zip(features, targets, weights):
            self.loss += mse_criterion(f, t) * w
        return self.loss


def gram(x):
    b = 1
    c, h, w = x.size()
    g = torch.bmm(x.reshape(b, c, h * w), x.reshape(b, c, h * w).transpose(1, 2))
    return g.div(h * w + eps)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.loss = 0

    def forward(self, features, targets, masks, weights=None):
        if weights is None:
            weights = [1 / len(features)] * len(features)

        loss_ = 0
        for f, t, m, w in zip(features, targets, masks, weights):
            m = m > 0
            if m.max():
                boxes = masks_to_boxes(m).to(dtype=torch.int)
                f = functional.crop(f, boxes[0][1], boxes[0][0], boxes[0][3] - boxes[0][1], boxes[0][2] - boxes[0][0])
                t = functional.crop(t, boxes[0][1], boxes[0][0], boxes[0][3] - boxes[0][1], boxes[0][2] - boxes[0][0])
            loss_ += mse_criterion(gram(f), gram(t)) * w
        self.loss = loss_
        return loss_
