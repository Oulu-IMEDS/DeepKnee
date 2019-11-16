"""
Main training script

(c) Aleksei Tiulpin, University of Oulu, 2017

"""

from __future__ import print_function

import argparse
import os
import gc
import pickle
import time

from termcolor import colored

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, mean_squared_error, cohen_kappa_score

from visdom import Visdom

cudnn.benchmark = True

from ouludeepknee.train.dataset import KneeGradingDataset, LimitedRandomSampler
from ouludeepknee.train.train_utils import train_epoch, adjust_learning_rate
from ouludeepknee.train.val_utils import validate_epoch
from ouludeepknee.train.model import KneeNet
from ouludeepknee.train.augmentation import (CenterCrop, CorrectGamma, Jitter,
                                             Rotate, CorrectBrightness, CorrectContrast)


SNAPSHOTS_KNEE_GRADING = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../snapshots_knee_grading'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data',  default='../../KL_data')
    parser.add_argument('--snapshots',  default=SNAPSHOTS_KNEE_GRADING)
    parser.add_argument('--experiment',  default='own_net')
    parser.add_argument('--patch_size', type=int, default=130)
    parser.add_argument('--base_width', type=int, default=32)
    parser.add_argument('--start_val', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_drop', type=int, default=20)
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=5e-5)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--val_bs', type=int, default=8)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--bootstrap', type=int, default=1)
    parser.add_argument('--n_batches', type=int, default=-1)
    parser.add_argument('--n_threads', type=int, default=20)
    parser.add_argument('--use_visdom', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    cur_lr = args.lr

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.isdir(args.snapshots):
        os.mkdir(args.snapshots)

    cur_snapshot = time.strftime('%Y_%m_%d_%H_%M_%S')
    os.mkdir(os.path.join(args.snapshots, cur_snapshot))
    with open(os.path.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Getting the name of the train and validation datasets
    # We oversample the train set
    train_cats_length = []
    for kl in range(5):
        train_cats_length.append(len(os.listdir(
            os.path.join(args.dataset, 'train', str(kl))
        )))

    oversample_size = int(sum(train_cats_length) / 5)
    train_files = []
    print(oversample_size)
    np.random.seed(args.seed)
    for kl in range(5):
        files = np.array(os.listdir(
            os.path.join(args.dataset, 'train', str(kl))
        ))
        train_files.extend(
            np.random.choice(
                files, size=oversample_size, replace=True
                ).tolist()
            )

    train_files = np.array(train_files)
    np.random.shuffle(train_files)
    val_files = np.array(os.listdir(os.path.join(args.dataset,'val')))
    
    if os.path.isfile(os.path.join(args.snapshots, 'mean_std.npy')):
        tmp = np.load(os.path.join(args.snapshots, 'mean_std.npy'))
        mean_vector, std_vector = tmp
    else:
        
        transf_tens= transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.float()
        ])
        
        train_ds = KneeGradingDataset(args.dataset, 
                                      train_files.tolist(), 
                                      transform=transf_tens,
                                      augment=CenterCrop(300),
                                      stage='train')
        
        train_loader = data.DataLoader(train_ds, batch_size=args.bs, num_workers=args.n_threads)

        mean_vector = np.zeros(1)
        std_vector = np.zeros(1)

        print(colored('==> ', 'green')+'Estimating the mean')
        pbar = tqdm(total=len(train_loader))
        for entry in train_loader:
            batch_l = entry[0]
            batch_m = entry[0]
            for j in range(mean_vector.shape[0]):
                mean_vector[j] += (batch_l[:, j, :, :].mean()+batch_l[:, j, :, :].mean())/2.
                std_vector[j] += (batch_l[:, j, :, :].std()+batch_m[:, j, :, :].std())/2.
            pbar.update()
        mean_vector /= len(train_loader)
        std_vector /= len(train_loader)
        np.save(os.path.join(args.snapshots, 'mean_std.npy'), [mean_vector, std_vector])
        pbar.close()
    print(colored('==> ', 'green')+'Mean: ', mean_vector)
    print(colored('==> ', 'green')+'Std: ', std_vector)

    # Defining the transforms
    # This is the transformation for each patch
    normTransform = transforms.Normalize(mean_vector, std_vector)
    patch_transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.float(),
            normTransform,
        ])

    # This we will use to globally augment the image
    augment_transforms = transforms.Compose([
        CorrectBrightness(0.7,1.3),
        CorrectContrast(0.7,1.3),
        Rotate(-15,15),
        CorrectGamma(0.5,2.5),
        Jitter(300, 6,20),       
    ])

    # Validation set
    val_ds = KneeGradingDataset(args.dataset, 
                                val_files.tolist(), 
                                transform=patch_transform,
                                augment=CenterCrop(300),
                                stage='val'
                                )

    val_loader = data.DataLoader(val_ds,
                                 batch_size=args.val_bs,
                                 num_workers=args.n_threads
                                 )

    print(colored('==> ', 'blue')+'Initialized the loaders....')

    # Network
    net = nn.DataParallel(KneeNet(args.base_width, args.drop, True))
    net.cuda()
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
    #                   lr=args.lr, weight_decay=args.wd, momentum=0.9)
    # Criterion
    criterion = F.cross_entropy
    # Visualizer-realted variables
    vis = Visdom()
    win = None
    win_metrics = None

    train_losses = []
    val_losses = []
    val_mse = []
    val_kappa = []
    val_acc = []

    best_dice = 0
    prev_model = None

    train_started = time.time()
    for epoch in range(args.n_epoch):

        # On each iteration we oversample the data to have everything correspond the torch implementation
        # This will be needed to oversample different KL-0 on each epoch
        train_files = []
        np.random.seed(args.seed)
        for kl in range(5):
            files = np.array(os.listdir(os.path.join(args.dataset,'train',str(kl))))
            train_files.extend(
                np.random.choice(
                    files, size=oversample_size*args.bootstrap, replace=True
                    ).tolist()
                )

        train_files = np.array(train_files)

        train_ds = KneeGradingDataset(args.dataset, 
                                      train_files.tolist(), 
                                      transform=patch_transform,
                                      augment=augment_transforms
                                      )
        N_batches = None
        if args.n_batches > 0:
            N_batches = args.n_batches

        if N_batches is not None:
            train_loader = data.DataLoader(train_ds, batch_size=args.bs,
                                           num_workers=args.n_threads,
                                           sampler=LimitedRandomSampler(train_ds, N_batches, args.bs)
                                           )
        else:
            train_loader = data.DataLoader(train_ds,
                                           batch_size=args.bs,
                                           num_workers=args.n_threads,
                                           shuffle=True
                                           )

        print(colored('==> ', 'blue')+'Epoch:', epoch+1, cur_snapshot)
        # Adjusting learning rate using the scheduler
        optimizer, cur_lr = adjust_learning_rate(optimizer, epoch+1, args)
        print(colored('==> ', 'red')+'LR:', cur_lr)
        # Training one epoch and measure the time
        start = time.time()
        train_loss = train_epoch(epoch, net, optimizer, train_loader, criterion, args.n_epoch)
        epoch_time = np.round(time.time() - start,4)
        print(colored('==> ', 'green')+'Epoch training time: {} s.'.format(epoch_time))
        # If it is time to start the validation, we will do it
        # args.args.start_val can be used to avoid time-consuming validation
        # in the beginning of the training
        if epoch >= args.start_val:
            start = time.time()
            val_loss, probs, truth, _ = validate_epoch(net, val_loader, criterion)

            preds = probs.argmax(1)
            # Validation metrics
            cm = confusion_matrix(truth, preds)
            kappa = np.round(cohen_kappa_score(truth, preds, weights="quadratic"),4)
            acc = np.round(np.mean(cm.diagonal().astype(float)/cm.sum(axis=1)),4)
            mse = np.round(mean_squared_error(truth, preds), 4)
            val_time = np.round(time.time() - start, 4)
            #Displaying the results
            print(colored('==> ', 'green')+'Kappa:', kappa)
            print(colored('==> ', 'green')+'Avg. class accuracy', acc)
            print(colored('==> ', 'green')+'MSE', mse)
            print(colored('==> ', 'green')+'Val loss:', val_loss)
            print(colored('==> ', 'green')+'Epoch val time: {} s.'.format(val_time))
            # Storing the logs
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_mse.append(mse)
            val_acc.append(acc)
            val_kappa.append(kappa)

        # Displaying the results in Visdom
        if epoch > args.start_val+1 and args.use_visdom:
            # Train/Val window
            if win is None:
                win = vis.line(
                    X=np.column_stack((np.arange(epoch, epoch+2),np.arange(epoch, epoch+2))),
                    Y=np.column_stack((np.array(train_losses[-2:]), np.array(val_losses[-2:]))),
                    opts=dict(title='[{}]\nTrain / val loss [{}]'.format(args.experiment, cur_snapshot),
                    legend=['Train', 'Validation'])
                )

            else:
                vis.line(
                    X=np.column_stack((np.arange(epoch, epoch+2),np.arange(epoch, epoch+2))),
                    Y=np.column_stack((np.array(train_losses[-2:]), np.array(val_losses[-2:]))),
                    win=win,
                    update='append'
                )
            # Metrics
            if win_metrics is None:
                win_metrics = vis.line(
                    X=np.column_stack((np.arange(epoch, epoch+2),np.arange(epoch, epoch+2),np.arange(epoch, epoch+2))),
                    Y=np.column_stack((1-np.array(val_mse[-2:]), np.array(val_kappa[-2:]),np.array(val_acc[-2:]))),
                    opts=dict(title='[{}]\nMetrics[{}]'.format(args.experiment, cur_snapshot),
                    legend=['1-MSE', 'Kappa','Accuracy'])
                )

            else:
                vis.line(
                    X=np.column_stack((np.arange(epoch, epoch+2),np.arange(epoch, epoch+2),np.arange(epoch, epoch+2))),
                    Y=np.column_stack((1-np.array(val_mse[-2:]), np.array(val_kappa[-2:]),np.array(val_acc[-2:]))),
                    win=win_metrics,
                    update='append'
                )

        # Making logs backup
        np.save(os.path.join(args.snapshots, cur_snapshot, 'logs.npy'), 
                [train_losses,val_losses, val_mse, val_acc, val_kappa])

        if epoch > args.start_val:
            # We will be saving only the snapshot which has lowest loss value on the validation set
            cur_snapshot_name = os.path.join(args.snapshots, cur_snapshot, 'epoch_{}.pth'.format(epoch+1))
            if prev_model is None:
                torch.save(net.state_dict(), cur_snapshot_name)
                prev_model = cur_snapshot_name
                best_kappa = kappa
            else:
                if kappa > best_kappa:
                    os.remove(prev_model)
                    best_kappa = kappa
                    print('Saved snapshot:',cur_snapshot_name)
                    torch.save(net.state_dict(), cur_snapshot_name)
                    prev_model = cur_snapshot_name

        gc.collect()


print(args.seed, 'Training took:', time.time()-train_started, 'seconds')
