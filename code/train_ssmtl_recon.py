import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

import numpy as np

from dataloader_torch import SSMTLModelDataset
from model_factory import ModelFactory

from dali_dataloader import masked_pipe, masked_arrow_pipe
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

RECON_BASED_MODELS = ['SSMTLAutoencoder','CNN3DResReconSkipV2','CNN3DResReconSSMTLDec']
MTL_BASED_MODELS = ['SSTMLAutoEncArrow']

class ReconLoss(nn.Module):
    def __init__(self, w1:float = 1.0, w2:float = 1.0, device='cuda:0'):
        super().__init__()
        self.loss_l1 = nn.L1Loss(reduction='mean')
        self.loss_l2 = nn.MSELoss(reduction='mean')
        self.loss_ssim = SSIM(data_range=1.0, reduction='elementwise_mean').to(device)
        self.w1 = w1
        self.w2 = w2

    def forward(self, pred, target):
        ssim_value = 1.0 - self.loss_ssim(pred, target)
        return self.w1 * self.loss_l1(pred, target) + self.w2 * self.loss_l2(pred, target) + ssim_value
    
class ReconArrowLoss(nn.Module):
    def __init__(self, w1:float = 1.0, w2:float = 1.0):
        super().__init__()
        self.loss_recon = ReconLoss(w1=w1, w2=w2)
        self.loss_arrow = nn.CrossEntropyLoss()

    def forward(self, pred_recon, target_recon, pred_arrow, target_arrow):
        loss = self.loss_recon(pred_recon, target_recon)
        loss += self.loss_arrow(pred_arrow, target_arrow.squeeze())
        return loss
    
def get_criterion(config):
    modelName = config['model']
    if modelName in RECON_BASED_MODELS:
        return ReconLoss()
    elif modelName in MTL_BASED_MODELS:
        return ReconArrowLoss()
    else:
        raise ValueError(f"Invalid model {modelName}")

def create_dali_loaders(config):   
    modelName = config['model']
    if modelName in RECON_BASED_MODELS:
        returnNames = ['sequence', 'masked_sequence', 'key', 'prefix']
        pipeline_func = masked_pipe
    elif modelName in MTL_BASED_MODELS:
        returnNames = ['sequence', 'masked_sequence', 'seq_backward', 'label_backward','key', 'prefix']
        pipeline_func = masked_arrow_pipe
    else:
        raise ValueError(f"Invalid model {modelName}")

    train_pipe = pipeline_func(ann_file=config['train_ann'],
                            num_frames=9,
                            batch_size=config['batch_size'],
                            num_threads=config['workers'],
                            train=True,
                            cover_factor=0.4, square_size=8)
    
    val_pipe = pipeline_func(ann_file=config['val_ann'],
                          num_frames=9,
                          batch_size=config['batch_size'],
                          num_threads=config['workers'],
                          train=True,
                          cover_factor=0.4, square_size=8)
    
    train_pipe.build()
    train_iter = DALIRaggedIterator(train_pipe, returnNames, size=-1)
    
    val_pipe.build()
    val_iter = DALIRaggedIterator(val_pipe, returnNames, size=-1)

    return train_iter, val_iter

def recon_step(model, batch, criterion):
    x, x_masked = batch[0]['sequence'], batch[0]['masked_sequence']
    y_recon = model(x_masked)
    loss = criterion(y_recon, x)
    
    return loss

def recon_arrow_step(model, batch, criterion):
    x, x_masked, x_arrow = batch[0]['sequence'], batch[0]['masked_sequence'], batch[0]['seq_backward']
    label_arrow = batch[0]['label_backward']
    y_recon, y_arrow = model(x_masked, x_arrow)

    loss = criterion(y_recon, x, y_arrow, label_arrow)

    return loss

def get_step_func(config):
    modelName = config['model']
    if modelName in RECON_BASED_MODELS:
        return recon_step
    elif modelName in MTL_BASED_MODELS:
        return recon_arrow_step
    else:
        raise ValueError(f"Invalid model {modelName}")


def train(config):
    device = set_device(use_gpu=config['use_gpu'])

    trainLoader, valLoader = create_dali_loaders(config)

    model = ModelFactory.create_model(model_name=config['model'],
                                      in_channel=config['in_channel'],
                                      out_channel=config['out_channel'])
    model = model.to(device=device)

    savedModel = config.get('saved_model', None)
    if savedModel:
        state_dict = torch.load(savedModel)
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
        print(f"Loaded saved model {savedModel}")

    criterion = get_criterion(config)
    optimizer = optim.Adam(lr=config['lr'], params=model.parameters())

    scheduler_config = config.get('scheduler_param', {})
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=scheduler_config['step_size'])

    step_func = get_step_func(config)

    start_time = time()
    best_val_loss = float('inf')

    writer = SummaryWriter(log_dir=f"runs/{config['model']}_{int(time())}")
    
    for epoch in range(config['num_epochs']):
        tic = time()
        
        loss_value = 0.0
        n_batches = 0

        t_train = tqdm(trainLoader, unit='batch')

        model.train()
        for batch in t_train:
            try:
                loss = step_func(model, batch, criterion)

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
        
        train_loss = loss_value / n_batches
        
        loss_value = 0.0
        n_batches = 0
        
        t_val = tqdm(valLoader, unit='batch')

        model.eval()
        for batch in t_val:
            try:
                with torch.no_grad():
                    loss = step_func(model, batch, criterion)
                
                loss_value += loss.detach().item()
                n_batches += 1

                desc = f"Epoch {epoch} Loss {loss_value/n_batches:.4f}"
                desc = f"{desc} Elapsed Time {time()-tic:.3f}"
                
                t_val.set_description(desc)

            except Exception as e:
                print(e)
                return
        
        scheduler.step()
        
        val_loss = loss_value / n_batches

        # Tensorboard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0] , epoch)

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