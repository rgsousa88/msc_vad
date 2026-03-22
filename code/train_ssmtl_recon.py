import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from dataloader_torch import SSMTLModelDataset
from models import SSTMLAutoEncoder

from dali_dataloader import masked_pipe
from nvidia.dali.plugin.pytorch import DALIRaggedIterator

from time import time
from tqdm import tqdm
import argparse

import os

from trainUtils import *
from configParser import ConfigParser


os.environ['DALI_DISABLE_NVML'] = '1'

torch.backends.cuda.matmul.allow_tf32 = True  # Usar TF32 na RTX 40
torch.backends.cudnn.allow_tf32 = True

def create_dali_loaders(config):   
    returnNames = ['sequence', 'masked_sequence', 'key', 'prefix']

    train_pipe = masked_pipe(ann_file=config['train_ann'],
                            num_frames=9,
                            batch_size=config['batch_size'],
                            num_threads=config['workers'],
                            train=True,
                            cover_factor=0.3, square_size=7)
    
    val_pipe = masked_pipe(ann_file=config['val_ann'],
                          num_frames=9,
                          batch_size=config['batch_size'],
                          num_threads=config['workers'],
                          train=True,
                          cover_factor=0.3, square_size=7)
    
    train_pipe.build()
    train_iter = DALIRaggedIterator(train_pipe, returnNames, size=-1)
    
    val_pipe.build()
    val_iter = DALIRaggedIterator(val_pipe, returnNames, size=-1)

    return train_iter, val_iter

def train(config):
    device = set_device(use_gpu=config['use_gpu'])

    trainLoader, valLoader = create_dali_loaders(config)

    model = SSTMLAutoEncoder(in_channel=config['in_channel'], out_channel=config['out_channel'])
    model = model.to(device=device)

    savedModel = config.get('saved_model', None)
    if savedModel:
        state_dict = torch.load(savedModel)
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
        print(f"Loaded saved model {savedModel}")

    loss_l1 = nn.L1Loss(reduction='mean')
    loss_l2 = nn.MSELoss(reduction='mean')

    optimizer = optim.Adam(lr=config['lr'], params=model.parameters())

    scheduler_config = config.get('scheduler_param', {})
    scheduler = get_scheduler(config['scheduler'], optimizer=optimizer, **scheduler_config)

    start_time = time()
    best_val_loss = float('inf')

    writer = SummaryWriter(log_dir=f"runs/ssmtl_recon_{int(time())}")
    
    for epoch in range(config['num_epochs']):
        tic = time()
        
        loss_value = 0.0
        n_batches = 0

        t_train = tqdm(trainLoader, unit='batch')

        model.train()
        for batch in t_train:
            try:
                x, x_masked = batch[0]['sequence'], batch[0]['masked_sequence']

                y_recon = model(x_masked)

                loss = loss_l1(x, y_recon)
                loss += loss_l2(x, y_recon)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value += loss.detach().item()
                n_batches += 1
                
                desc = f"Epoch {epoch} Loss {loss_value/n_batches:.4f}"
                desc = f"{desc} Elapsed Time {time()-tic:.3f}"
                
                t_train.set_description(desc)

            except Exception as e:
                print(f"Exception {e}")
                return
        
        scheduler.step()
        train_loss = loss_value / n_batches
        
        loss_value = 0.0
        n_batches = 0
        
        t_val = tqdm(valLoader, unit='batch')

        model.eval()
        for batch in t_val:
            try:
                with torch.no_grad():
                    x, x_masked = batch[0]['sequence'], batch[0]['masked_sequence']

                    y_recon = model(x_masked)

                    loss = loss_l1(x, y_recon)
                    loss += loss_l2(x, y_recon)
                
                loss_value += loss.detach().item()
                n_batches += 1

                desc = f"Epoch {epoch} Loss {loss_value/n_batches:.4f}"
                desc = f"{desc} Elapsed Time {time()-tic:.3f}"
                
                t_val.set_description(desc)

            except Exception as e:
                print(e)
                return
        
        val_loss = loss_value / n_batches

        # Tensorboard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            saveModel(config=config, model=model, epoch=epoch, val_loss=val_loss, val_acc=0.0)

        epochReport = f"Train Loss {train_loss:.4f} Val Loss {val_loss:.4f}\n"
        print(f"Epoch {epoch} LR {optimizer.param_groups[0]['lr']:.4f} {epochReport}")
        print(f"Elapsed Time {time() - start_time:.4f}")

    writer.close()
    print(f"Total time {time() - start_time:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ShanghaiTech SSMTL')

    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()

    config = ConfigParser(args.config_file).config

    train(config)