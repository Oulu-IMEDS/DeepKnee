"""

Implementation of the network architecture proposed by Antony et al.

(c) Aleksei Tiulpin, University of Oulu, 2017

"""

import torch.nn as nn
import torch.nn.functional as F


def ConvBlockK(inp, out, K, stride, pad):
    """
    KxK ConvNet building block with different activations support.
    
    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
    
    """
    return nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=K, stride=stride, padding=pad),
        nn.BatchNorm2d(out),
        nn.ReLU(inplace=True)
    )


class AntonyNet2Heads(nn.Module):
    """
    Net with two heads.

    """
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlockK(1, 32, 11, 2, 0)
        
        self.block2 = nn.Sequential(ConvBlockK(32, 64, 3, 1, 1), 
                                    ConvBlockK(64, 64, 3, 1, 1))
        
        self.block3 = nn.Sequential(ConvBlockK(64, 96, 3, 1, 1),
                                    nn.Dropout(0.5),
                                    
                                    ConvBlockK(96, 96, 3, 1, 1),
                                    nn.Dropout(0.5))
        
        self.fc = nn.Sequential(nn.Linear(17952, 512),
                                nn.Dropout(0.5))
        
        self.fc_clf = nn.Linear(512, 5)
        self.fc_reg = nn.Linear(512, 1)
        
    def forward(self, x):
        o = F.max_pool2d(self.block1(x), 3, stride=2)
        o = F.max_pool2d(self.block2(o), 3, stride=2)
        o = F.max_pool2d(self.block3(o), 3, stride=2)
        
        o = self.fc(o.view(x.size(0),-1))
        
        return self.fc_clf(o), self.fc_reg(o)
