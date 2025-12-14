import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

import base_dataloader as bd
from hibrid_mask_dataloader import HybridMaskedVideoIterator
from models import CNN3DRecon

from time import time
from tqdm import tqdm
import argparse

import os

from trainUtils import *
from configParser import ConfigParser

os.environ['DALI_DISABLE_NVML'] = '1'

def recon_accuracy(pred, gt, threshold=1e-2):
    diff = torch.abs(F.sigmoid(pred) - gt)
    numElem = gt.numel()
    totalSim = diff.le(threshold).sum().detach()

    return totalSim / numElem

def train(config):
    device = set_device(use_gpu=config['use_gpu'])

    recon_config = config.get('recon_param', {})
    shape = (config['input_size'][0], config['input_size'][1])

    trainPipe = bd.video_pipe(file_root=config['trainPath'], train=True, shape=shape,
                          sequence_length=config['seqLen'], stride=config['fstride'], step=config['cstride'],
                          batch_size=config['batch_size'],
                          change_color_prob=0.5)
    trainLoader = HybridMaskedVideoIterator(pipeline=trainPipe,
                                            input_shape=shape,
                                            sequence_length=config['seqLen'],
                                            batch_size=config['batch_size'],
                                            square_size=recon_config["square_size"],
                                            mask_type=recon_config["mask_type"],
                                            mask_prob=0.95)

    valPipe = bd.video_pipe(file_root=config['valPath'], train=False, shape=shape,
                          sequence_length=config['seqLen'], stride=config['fstride'], step=config['cstride'],
                          batch_size=config['batch_size'])

    valLoader = HybridMaskedVideoIterator(pipeline=valPipe,
                                        input_shape=shape,
                                        sequence_length=config['seqLen'],
                                        batch_size=config['batch_size'],
                                        square_size=recon_config["square_size"],
                                        mask_type=recon_config["mask_type"],
                                        mask_prob=0.95)

    model = CNN3DRecon()
    model = model.to(device=device)

    savedModel = config.get('saved_model', None)
    if savedModel:
        state_dict = torch.load(savedModel)
        model.load_state_dict(state_dict['model_state_dict'],strict=False)
        print(f"Loaded saved model {savedModel}")


    criterion = get_losses(config['loss'])
    optimizer = get_optmizer(optimizer_name=config['optmizer'], params=model.parameters(), lr=config['lr'])
    
    scheduler_config = config.get('scheduler_param', {})
    scheduler = get_scheduler(config['scheduler'], optimizer=optimizer, **scheduler_config)

    start_time = time()
    best_val_acc = 0.0
    MIN_VAL_ACC = 0.10
    
    for epoch in range(config['num_epochs']):
        tic = time()
        
        loss_value = 0.0
        n_batches = 0
        acc_value = 0.0

        t_train = tqdm(trainLoader, unit='batch')

        model.train()
        for batch in t_train:
            try:
                masked_videos, original_videos,_ = batch
                loss, metric = trainStep(model=model, data=masked_videos, target=original_videos, 
                                         criterion=criterion, optimizer=optimizer, evalFunc=recon_accuracy)
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
                masked_videos, original_videos,_ = batch
                loss,metric = valStep(model=model, data=masked_videos, target=original_videos,
                                      criterion=criterion, evalFunc=recon_accuracy)
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


