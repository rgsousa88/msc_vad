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
from models import SSMTLModel

from dali_dataloader import ssmtl_pipe
from nvidia.dali.plugin.pytorch import DALIRaggedIterator

from time import time
from tqdm import tqdm
import argparse

import os

from trainUtils import *
from configParser import ConfigParser

os.environ['DALI_DISABLE_NVML'] = '1'


def create_torch_loaders(config, prefetch_factor=4):
    trainDataset = SSMTLModelDataset(annotation_path=config['train_ann'],
                                     input_size=config['input_size'],
                                     window=config['window'],
                                     transform=None)
    
    trainLoader = DataLoader(trainDataset,
                             batch_size=config['batch_size'],
                             shuffle=True,
                             num_workers=config['workers'],
                             prefetch_factor=prefetch_factor,  # Pré-carregar 4 batches
                             persistent_workers=True,  # Manter workers vivos entre épocas
                             pin_memory=True,  # Acelera transferência CPU->GPU
                             pin_memory_device='cuda',  # Direto para GPU
                             drop_last=True)


    valDataset = SSMTLModelDataset(annotation_path=config['val_ann'],
                                  input_size=config['input_size'],
                                  window=config['window'])
    
    valLoader = DataLoader(valDataset,
                           batch_size=config['batch_size'],
                           shuffle=True,
                           num_workers=config['workers'],
                           prefetch_factor=prefetch_factor,  # Pré-carregar 4 batches
                           persistent_workers=True,  # Manter workers vivos entre épocas
                           pin_memory=True,  # Acelera transferência CPU->GPU
                           pin_memory_device='cuda',  # Direto para GPU
                           drop_last=True)
    
    return trainLoader, valLoader

def create_dali_loaders(config):   
    returnNames = ['x_arrow', 'l_arrow',
                   'x_motion', 'l_motion',
                   'x_recon', 'x_distil',
                   'feat_distil',
                   'key','prefix']

    train_pipe = ssmtl_pipe(ann_file=config['train_ann'],
                            num_frames=31,
                            batch_size=config['batch_size'],
                            num_threads=config['workers'],
                            train=True)
    
    val_pipe = ssmtl_pipe(ann_file=config['val_ann'],
                          num_frames=31,
                          batch_size=config['batch_size'],
                          num_threads=config['workers'],
                          train=True)
    
    train_pipe.build()
    train_iter = DALIRaggedIterator(train_pipe, returnNames, size=-1)
    
    val_pipe.build()
    val_iter = DALIRaggedIterator(val_pipe, returnNames, size=-1)

    return train_iter, val_iter

def train(config):
    device = set_device(use_gpu=config['use_gpu'])
    trainLoader, valLoader = create_dali_loaders(config)

    model = SSMTLModel(in_channel=config['in_channel'], out_channel=config['out_channel'])
    model = model.to(device=device)

    savedModel = config.get('saved_model', None)
    if savedModel:
        state_dict = torch.load(savedModel)
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
        print(f"Loaded saved model {savedModel}")

    loss_arrow = nn.CrossEntropyLoss()
    loss_motion = nn.CrossEntropyLoss()
    loss_recon = nn.L1Loss(reduction='mean')
    loss_distill = nn.L1Loss(reduction='mean')

    optimizer = optim.Adam(lr=0.001, params=model.parameters())

    #scheduler_config = config.get('scheduler_param', {})
    #scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=scheduler_config['step_size'])

    start_time = time()
    best_val_loss = float('inf')

    writer = SummaryWriter(log_dir=f"runs/{config['model']}_{int(time())}")
    
    for epoch in range(config['num_epochs']):
        tic = time()
        
        train_loss_recon = 0.0
        train_loss_arrow = 0.0
        train_loss_motion = 0.0
        train_loss_distill = 0.0
        train_acc_arrow = 0.0
        train_acc_motion = 0.0
        loss_value = 0.0
        n_batches = 0

        t_train = tqdm(trainLoader, unit='batch')

        model.train()
        for batch in t_train:
            try:
                x_arrow, l_arrow = batch[0]['x_arrow'], batch[0]['l_arrow']
                x_motion, l_motion = batch[0]['x_motion'], batch[0]['l_motion']
                x_recon, x_distil, feat_distil = batch[0]['x_recon'], batch[0]['x_distil'], batch[0]['feat_distil']

                y_arrow, y_motion, y_recon, y_distil = model(x_arrow, x_motion, x_recon, x_distil)

                loss_r = loss_recon(y_recon, x_distil[:,:,3,:,:].reshape(y_recon.shape))
                loss_a = loss_arrow(y_arrow, l_arrow.squeeze())
                loss_m = loss_motion(y_motion, l_motion.squeeze())
                loss_d = loss_distill(y_distil, feat_distil)

                loss = loss_r + loss_a + loss_m + 0.2 * loss_d

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred_arrow = torch.argmax(y_arrow, dim=1)
                pred_motion = torch.argmax(y_motion, dim=1)
                acc_arrow = (pred_arrow == l_arrow.squeeze()).float().mean()
                acc_motion = (pred_motion == l_motion.squeeze()).float().mean()
                
                train_loss_recon += loss_r.detach().item()
                train_loss_arrow += loss_a.detach().item()
                train_loss_motion += loss_m.detach().item()
                train_loss_distill += loss_d.detach().item()
                train_acc_arrow += acc_arrow.detach().item()
                train_acc_motion += acc_motion.detach().item()

                loss_value += loss.detach().item()
                n_batches += 1
                
                desc = f"Epoch {epoch} Loss {loss_value/n_batches:.4f}"
                desc = f"{desc} Elapsed Time {time()-tic:.3f}"
                
                t_train.set_description(desc)

            except Exception as e:
                print(f"Exception {e}")
                return
        
        #scheduler.step()
        train_loss = loss_value / n_batches
        train_loss_recon /= n_batches
        train_loss_arrow /= n_batches
        train_loss_motion /= n_batches
        train_loss_distill /= n_batches
        train_acc_arrow /= n_batches
        train_acc_motion /= n_batches
        
        val_loss_recon = 0.0
        val_loss_arrow = 0.0
        val_loss_motion = 0.0
        val_loss_distill = 0.0
        val_acc_arrow = 0.0
        val_acc_motion = 0.0
        loss_value = 0.0
        n_batches = 0
        
        t_val = tqdm(valLoader, unit='batch')

        model.eval()
        for batch in t_val:
            try:
                with torch.no_grad():
                    x_arrow, l_arrow = batch[0]['x_arrow'], batch[0]['l_arrow']
                    x_motion, l_motion = batch[0]['x_motion'], batch[0]['l_motion']
                    x_recon, x_distil, feat_distil = batch[0]['x_recon'], batch[0]['x_distil'], batch[0]['feat_distil']

                    y_arrow, y_motion, y_recon, y_distil = model(x_arrow, x_motion, x_recon, x_distil)
                    
                    loss_r = loss_recon(y_recon, x_distil[:,:,3,:,:].reshape(y_recon.shape))
                    loss_a = loss_arrow(y_arrow, l_arrow.squeeze())
                    loss_m = loss_motion(y_motion, l_motion.squeeze())
                    loss_d = loss_distill(y_distil, feat_distil)

                    loss = loss_r + loss_a + loss_m + 0.2 * loss_d

                    pred_arrow = torch.argmax(y_arrow, dim=1)
                    pred_motion = torch.argmax(y_motion, dim=1)
                    acc_arrow = (pred_arrow == l_arrow.squeeze()).float().mean()
                    acc_motion = (pred_motion == l_motion.squeeze()).float().mean()
                    
                    val_loss_recon += loss_r.detach().item()
                    val_loss_arrow += loss_a.detach().item()
                    val_loss_motion += loss_m.detach().item()
                    val_loss_distill += loss_d.detach().item()
                    val_acc_arrow += acc_arrow.detach().item()
                    val_acc_motion += acc_motion.detach().item()
                
                loss_value += loss.detach().item()
                n_batches += 1

                desc = f"Epoch {epoch} Loss {loss_value/n_batches:.4f}"
                desc = f"{desc} Elapsed Time {time()-tic:.3f}"
                
                t_val.set_description(desc)

            except Exception as e:
                print(e)
                return
        
        val_loss = loss_value / n_batches
        val_loss_recon /= n_batches
        val_loss_arrow /= n_batches
        val_loss_motion /= n_batches
        val_loss_distill /= n_batches
        val_acc_arrow /= n_batches
        val_acc_motion /= n_batches

        # Tensorboard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_recon", train_loss_recon, epoch)
        writer.add_scalar("Loss/train_arrow", train_loss_arrow, epoch)
        writer.add_scalar("Loss/train_motion", train_loss_motion, epoch)
        writer.add_scalar("Loss/train_distill", train_loss_distill, epoch)
        writer.add_scalar("Accuracy/train_arrow", train_acc_arrow, epoch)
        writer.add_scalar("Accuracy/train_motion", train_acc_motion, epoch)

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Loss/val_recon", val_loss_recon, epoch)
        writer.add_scalar("Loss/val_arrow", val_loss_arrow, epoch)
        writer.add_scalar("Loss/val_motion", val_loss_motion, epoch)
        writer.add_scalar("Loss/val_distill", val_loss_distill, epoch)
        writer.add_scalar("Accuracy/val_arrow", val_acc_arrow, epoch)
        writer.add_scalar("Accuracy/val_motion", val_acc_motion, epoch)
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            saveModel(config=config, model=model, epoch=epoch, val_loss=val_loss, val_acc=0.0)

        epochReport = f"Train Loss {train_loss:.4f} Val Loss {val_loss:.4f}\n"
        print(f"acc arrow train = {train_acc_arrow:.4f}, val = {val_acc_arrow:.4f}")
        print(f"acc motion train = {train_acc_motion:.4f}, val = {val_acc_motion:.4f}")
        print(f"loss_resnet = {train_loss_distill:.4f}, loss_recon = {train_loss_recon:.4f}")
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