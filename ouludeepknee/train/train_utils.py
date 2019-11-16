"""
This file contains the training utils

(c) Aleksei Tiulpin, University of Oulu, 2017
"""

from __future__ import print_function

import gc

import torch
from torch.autograd import Variable


def adjust_learning_rate(optimizer, epoch, args):
    """
    Decreases the initial LR by 10 every drop_step epochs. 
    Conv layers learn slower if specified in the optimizer.
    """
    lr = args.lr * (0.1 ** (epoch // args.lr_drop))
    if lr < args.lr_min:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr


def train_epoch(epoch, net, optimizer, train_loader, criterion, max_ep):

    net.train(True)

    running_loss = 0.0
    n_batches = len(train_loader)
    for i, (batch_l, batch_m, targets, names) in enumerate(train_loader):
        optimizer.zero_grad()

        # forward + backward + optimize
        labels = Variable(targets.long().cuda())
        inputs_l = Variable(batch_l.cuda())
        inputs_m = Variable(batch_m.cuda())
        
        outputs = net(inputs_l, inputs_m)
        
        if batch_l.size(0) != torch.cuda.device_count():
            outputs = outputs.squeeze()
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        print('[%d | %d, %5d / %d] | Running loss: %.3f / loss %.3f' %
              (epoch + 1, max_ep, i + 1, n_batches, running_loss / (i+1), loss.data[0]))
        gc.collect()
    gc.collect()

    return running_loss/n_batches
