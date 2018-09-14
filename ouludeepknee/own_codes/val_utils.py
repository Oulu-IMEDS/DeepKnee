"""
Validation utils

(c) Aleksei Tiulpin, University of Oulu, 2017
"""

import gc
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def validate_epoch(net, val_loader, criterion):

    net.train(False)

    running_loss = 0.0
    n_batches = len(val_loader)
    sm = nn.Softmax()
    
    truth = []
    preds = []
    bar = tqdm(total=len(val_loader),desc='Processing', ncols=90)
    names_all = []
    for i, (batch_l, batch_m, targets, names) in enumerate(val_loader):
        labels = Variable(targets.long().cuda())
        
        inputs_l = Variable(batch_l.cuda())
        inputs_m = Variable(batch_m.cuda())
        
        outputs = net(inputs_l, inputs_m)
        
        if batch_l.size(0) != torch.cuda.device_count():
            outputs = outputs.squeeze()
        
        loss = criterion(outputs, labels)
        probs = sm(outputs).data.cpu().numpy()
        preds.append(probs)
        truth.append(targets.cpu().numpy())
        names_all.extend(names)

        running_loss += loss.data[0]
        bar.update(1)
        gc.collect()
    gc.collect()
    bar.close()
    preds = np.vstack(preds)
    truth = np.hstack(truth)
    
    return running_loss/n_batches, preds, truth, names_all
