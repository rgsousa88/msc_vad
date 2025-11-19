import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

import base_dataloader as bd
from models import CNN3D, CameraClassifier

from nvidia.dali.plugin.pytorch import DALIGenericIterator

from time import time
from tqdm import tqdm
import argparse

import os

from trainUtils import *
from configParser import ConfigParser

os.environ['DALI_DISABLE_NVML'] = '1'

def topKAccuracy(pred, targets, topk=1):
    assert pred.dim() == 2
    assert targets.dim() == 1
    assert pred.size(0) == targets.size(0)
    
    with torch.no_grad():
        if isinstance(topk, int):
            topk = (topk,)
        
        maxk = max(topk)
        batch_size = targets.size(0)
        
        # Top-k
        _, pred = pred.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        result = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            result[f'top{k}'] = (correct_k / batch_size).item()
        
        return result if len(result) > 1 else result[f'top{topk[0]}']

def train(config):
    device = set_device(use_gpu=config['use_gpu'])

    shape = (config['input_size'][0], config['input_size'][1])
    trainPipe = bd.video_pipe(file_root=config['trainPath'], train=True, shape=shape,
                          sequence_length=config['seqLen'], stride=config['fstride'], step=config['cstride'],
                          batch_size=config['batch_size'])
    trainPipe.build()
    trainLoader = DALIGenericIterator(trainPipe, ['videos','labels'], reader_name='seq')

    valPipe = bd.video_pipe(file_root=config['valPath'], train=False, shape=shape,
                          sequence_length=config['seqLen'], stride=config['fstride'], step=config['cstride'],
                          batch_size=config['batch_size'])
    valPipe.build()
    valLoader = DALIGenericIterator(valPipe, ['videos','labels'], reader_name='seq')

    model = CameraClassifier(n_classes=config['num_classes'], temp_pool=config['tempPool'])
    model = model.to(device=device)

    criterion = get_losses(config['loss'])
    optimizer = get_optmizer(optimizer_name=config['optmizer'], params=model.parameters(), lr=config['lr'])
    
    scheduler_config = config.get('scheduler_param', {})
    scheduler = get_scheduler(config['scheduler'], optimizer=optimizer, **scheduler_config)

    start_time = time()
    best_val_acc = 0.0
    MIN_VAL_ACC = 0.50
    
    for epoch in range(config['num_epochs']):
        tic = time()
        
        loss_value = 0.0
        n_batches = 0
        acc_value = 0.0

        t_train = tqdm(trainLoader, unit='batch')

        model.train()
        for batch in t_train:
            try:
                loss, metric = trainStep(model=model, data=batch[0]['videos'], target=batch[0]['labels'], 
                                         criterion=criterion, optimizer=optimizer, evalFunc=topKAccuracy)
                loss_value += loss
                acc_value += metric
                n_batches += 1
                
                desc = f"Epoch {epoch} Loss {loss_value/n_batches:.4f}"
                desc = f"{desc} Acc {acc_value/n_batches:.4f} "
                desc = f"{desc} Elapsed Time {time()-tic:.3f}"
                
                t_train.set_description(desc)

            except Exception as e:
                print(e)
                return
        
        scheduler.step()
        train_loss = loss_value / n_batches
        train_acc = acc_value / n_batches
        
        loss_value = 0.0
        acc_value = 0.0
        n_batches = 0
        
        t_val = tqdm(valLoader, unit='batch')

        model.eval()
        for batch in t_val:
            try:
                loss,metric = valStep(model=model, data=batch[0]['videos'], target=batch[0]['labels'],
                                      criterion=criterion, evalFunc=topKAccuracy)
                loss_value += loss
                acc_value += metric
                n_batches += 1

                desc = f"Epoch {epoch} Loss {loss_value/n_batches:.4f}"
                desc = f"{desc} Acc {acc_value/n_batches:.4f} "
                desc = f"{desc} Elapsed Time {time()-tic:.3f}"
                
                t_val.set_description(desc)

            except Exception as e:
                print(e)
                return
        
        val_loss = loss_value / n_batches
        val_acc = acc_value / n_batches

        if val_acc > best_val_acc and val_acc > MIN_VAL_ACC:
            diffAcc = abs(val_acc - best_val_acc)
            best_val_acc = val_acc
            saveModel(config=config, model=model, epoch=epoch, val_loss=val_loss, val_acc=val_acc)
            if diffAcc < 1e-4:
                break

        epochReport = f"Train Loss {train_loss:.4f} Val Loss {val_loss:.4f}\nTrain Acc {train_acc:.4f} Val Acc {val_acc:.4f}"
        print(f"Epoch {epoch} LR {optimizer.param_groups[0]['lr']:.4f} {epochReport}")
        print(f"Elapsed Time {time() - start_time:.4f}")

    print(f"Total time {time() - start_time:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ShanghaiTech CamClassifier Self-Sup')

    parser.add_argument('--config_file', type=str, required=True, help='Seu nome')
    args = parser.parse_args()

    config = ConfigParser(args.config_file).config

    train(config)


