"""
Network architecture class

Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBlock3(inp, out, stride, pad):
    """
    3x3 ConvNet building block with different activations support.
    
    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """
    return nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=3, stride=stride, padding=pad),
        nn.BatchNorm2d(out, eps=1e-3),
        nn.ReLU(inplace=True)
    )


def weights_init_uniform(m):
    """
    Initializes the weights using kaiming method.

    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)


class Branch(nn.Module):
    def __init__(self, bw):
        super().__init__()
        self.block1 = nn.Sequential(ConvBlock3(1, bw, 2, 0),
                                    ConvBlock3(bw, bw, 1, 0),
                                    ConvBlock3(bw, bw, 1, 0),
                                    )

        self.block2 = nn.Sequential(ConvBlock3(bw, bw * 2, 1, 0),
                                    ConvBlock3(bw * 2, bw * 2, 1, 0),
                                    )

        self.block3 = ConvBlock3(bw * 2, bw * 4, 1, 0)

    def forward(self, x):
        o1 = F.max_pool2d(self.block1(x), 2)
        o2 = F.max_pool2d(self.block2(o1), 2)
        return F.avg_pool2d(self.block3(o2), 10).view(x.size(0), -1)


def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val


class KneeNet(nn.Module):
    """
    Siamese Net to automatically grade osteoarthritis 
    
    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """

    def __init__(self, bw, drop, use_w_init=True):
        super().__init__()
        self.branch = Branch(bw)

        if drop > 0:
            self.final = nn.Sequential(nn.Dropout(p=drop), nn.Linear(2 * bw * 4, 5))
        else:
            self.final = nn.Linear(2 * bw * 4, 5)

        # Custom weights initialization
        if use_w_init:
            self.apply(weights_init_uniform)

    def forward(self, x1, x2):
        # Shared weights
        o1 = self.branch(x1)
        o2 = self.branch(x2)
        feats = torch.cat([o1, o2], 1)

        return self.final(feats)
