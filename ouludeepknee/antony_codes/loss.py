"""
Implementation of the loss proposed by Antony et al.

"""

import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, ratio=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.ratio = ratio
        
    def forward(self, outputs_clf, outputs_reg, labels):
        return self.ratio*self.ce(outputs_clf, labels)+(1-self.ratio)*self.mse(outputs_reg, labels.float())
