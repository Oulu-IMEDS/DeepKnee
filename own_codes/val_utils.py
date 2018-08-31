"""
Validation utils

(c) Aleksei Tiulpin, University of Oulu, 2017
"""

import gc
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import torch
import os


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


def validate_epoch_tta(net, val_loader, criterion, save_fld=None):
    softmax = nn.Sigmoid()
    running_loss = 0
    net.train(False)
    bar = Bar('Processing', max=len(val_loader))

    dice_sum = 0
    for i, entry in enumerate(val_loader):

        batches = entry[:-2]
        targets = entry[-2]
        names = entry[-1]
        cur_preds = np.zeros(targets.size(),dtype=np.float32)

        for aug_number, batch in enumerate(batches):

            inputs = Variable(batch.cuda(), volatile=True)
            outputs = net(inputs)
            if batch.size(0) != torch.cuda.device_count():
                outputs = outputs.squeeze()

            if aug_number == 1:
                preds_cpu = softmax(outputs.data).cpu().numpy()
                for pic_num in range(preds_cpu.shape[0]):
                    cur_preds[pic_num] += cv2.flip(preds_cpu[pic_num], 1)
            else:
                cur_preds += softmax(outputs.data).cpu().numpy()

            del inputs
            del outputs
            gc.collect()

        cur_preds /= len(batches)

        labels = Variable(targets.float().cuda(), volatile=True)
        averaged = Variable(torch.from_numpy(cur_preds).cuda(), volatile=True)
        loss = criterion(averaged, labels)
        running_loss += loss.data[0]
        tmp = 0
        for j in range(cur_preds.shape[0]):
            img_j = cur_preds[j, :].reshape((cur_preds.shape[-2], cur_preds.shape[-1]))
            if save_fld is not None:
                mask_big = img_j[0:SCALE_HEIGHT, 0:SCALE_WIDTH]
                mask_save = np.uint8(mask_big*255)
                cv2.imwrite(os.path.join(save_fld, names[j][:-4]+'.png'), mask_save)

            ground_truth_j = targets[j, :].squeeze().numpy()
            mask = img_j > 0.5
            tmp += dice_score(mask, ground_truth_j)

        dice_sum += tmp/batch.size(0)
        gc.collect()
        bar.next()
    bar.finish()

    return running_loss/len(val_loader), dice_sum/len(val_loader)

