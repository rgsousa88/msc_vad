import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import numpy as np

from dataloader_torch import SSMTLModelDataset
from models import SSMTLModel

from time import time
from tqdm import tqdm
import argparse

import os

from trainUtils import *
from configParser import ConfigParser

from model_factory import ModelFactory

os.environ['DALI_DISABLE_NVML'] = '1'


def train(config):
    device = set_device(use_gpu=config['use_gpu'])

    transfTrain = [v2.ColorJitter(brightness=.5, contrast=.5, hue=.3),
                   v2.RandomHorizontalFlip(p=0.5)]

    trainDataset = SSMTLModelDataset(annotation_path=config['trainAnn'],
                                     input_size=config['input_size'],
                                     window=config['window'],
                                     transform=transfTrain)
    
    trainLoader = DataLoader(trainDataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)


    valDataset = SSMTLModelDataset(annotation_path=config['valAnn'],
                                  input_size=config['input_size'],
                                  window=config['window'])
    
    valLoader = DataLoader(valDataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    model = SSMTLModel(in_channel=config['in_channel'], out_channel=config['out_channel'])
    model = model.to(device=device)

    savedModel = config.get('saved_model', None)
    if savedModel:
        state_dict = torch.load(savedModel)
        model.load_state_dict(state_dict['model_state_dict'], strict=True)
        print(f"Loaded saved model {savedModel}")

    loss_arrow = nn.CrossEntropyLoss()
    loss_motion = nn.CrossEntropyLoss()
    loss_recon = nn.L1Loss(reduction='mean')
    loss_distill = nn.L1Loss(reduction='mean')

    optimizer = optim.Adam(lr=0.01, params=model.parameters())

    start_time = time()
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        tic = time()
        
        loss_value = 0.0
        n_batches = 0

        t_train = tqdm(trainLoader, unit='batch')

        model.train()
        for batch in t_train:
            try:
                (x_arrow, l_arrow), (x_motion, l_motion), x_recon, (x_distil, feat_distil) = batch
                
                x_arrow = x_arrow.to(device)
                l_arrow = l_arrow.to(device)
                x_motion = x_motion.to(device)
                l_motion = l_motion.to(device)
                x_recon = x_recon.to(device)
                x_distil = x_distil.to(device)
                feat_distil = feat_distil.to(device)

                y_arrow, y_motion, y_recon, y_distil = model(x_arrow, x_motion, x_recon, x_distil)

                loss = loss_arrow(y_arrow, l_arrow)
                loss += loss_motion(y_motion, l_motion)
                loss += loss_recon(y_recon, x_distil.reshape(y_recon.shape))
                loss += 0.2 * loss_distill(y_distil, feat_distil)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value += loss
                n_batches += 1
                
                desc = f"Epoch {epoch} Loss {loss_value/n_batches:.4f}"
                desc = f"{desc} Elapsed Time {time()-tic:.3f}"
                
                t_train.set_description(desc)

            except Exception as e:
                print(e)
                return
        
        train_loss = loss_value / n_batches
        
        loss_value = 0.0
        n_batches = 0
        
        t_val = tqdm(valLoader, unit='batch')

        model.eval()
        for batch in t_val:
            try:
                with torch.no_grad():
                    (x_arrow, l_arrow), (x_motion, l_motion), x_recon, (x_distil, feat_distil) = batch

                    x_arrow = x_arrow.to(device)
                    l_arrow = l_arrow.to(device)
                    x_motion = x_motion.to(device)
                    l_motion = l_motion.to(device)
                    x_recon = x_recon.to(device)
                    x_distil = x_distil.to(device)
                    feat_distil = feat_distil.to(device)

                    y_arrow, y_motion, y_recon, y_distil = model(x_arrow, x_motion, x_recon, x_distil)
                    
                    loss = loss_arrow(y_arrow, l_arrow)
                    loss += loss_motion(y_motion, l_motion)
                    loss += loss_recon(y_recon, x_distil.reshape(y_recon.shape))
                    loss += 0.2 * loss_distill(y_distil, feat_distil)
                
                loss_value += loss
                n_batches += 1

                desc = f"Epoch {epoch} Loss {loss_value/n_batches:.4f}"
                desc = f"{desc} Elapsed Time {time()-tic:.3f}"
                
                t_val.set_description(desc)

            except Exception as e:
                print(e)
                return
        
        val_loss = loss_value / n_batches

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            saveModel(config=config, model=model, epoch=epoch, val_loss=val_loss, val_acc=0.0)

        epochReport = f"Train Loss {train_loss:.4f} Val Loss {val_loss:.4f}\n"
        print(f"Epoch {epoch} LR {optimizer.param_groups[0]['lr']:.4f} {epochReport}")
        print(f"Elapsed Time {time() - start_time:.4f}")

    print(f"Total time {time() - start_time:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ShanghaiTech SSMTL')

    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()

    config = ConfigParser(args.config_file).config

    train(config)