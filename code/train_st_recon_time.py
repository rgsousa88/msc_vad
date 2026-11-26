import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from model_factory import ModelFactory

from mask_dali_dataloader import masked_time_pipe
from nvidia.dali.plugin.pytorch import DALIRaggedIterator

from time import time
from tqdm import tqdm
import argparse

import os

from trainUtils import *
from configParser import ConfigParser

ST_RECON_TIME_MODEL = ['STReconTime','CNN3DReconTimeAttention']

set_seed(12334)

class ReconArrowLoss(nn.Module):
    """Reconstruction + temporal-order loss with uncertainty-based task weighting.

    Task balancing follows Kendall et al., "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics" (CVPR 2018). Each task gets
    a learnable log-variance ``s = log(sigma^2)`` and the loss uses homoscedastic
    uncertainty weighting:

        reconstruction (regression, L2/Gaussian): 0.5 * exp(-s_recon) * L_recon + 0.5 * s_recon
        temporal order (classification):          1.0 * exp(-s_time)  * L_time  + 0.5 * s_time

    The ``log_var_*`` tensors are ``nn.Parameter``s, so they must be optimised:
    ``train()`` adds ``criterion.parameters()`` to the optimizer. Set
    ``uncertainty_weighting=False`` to fall back to fixed ``w1``/``w2`` weights.
    """
    def __init__(self, w1:float = 1.0, w2:float = 1.0, device='cuda:0',
                 uncertainty_weighting:bool = True):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.uncertainty_weighting = uncertainty_weighting
        # log(sigma^2) per task, initialised so that exp(-s) = 1 (sigma = 1).
        self.log_var_recon = nn.Parameter(torch.zeros(()))
        self.log_var_time = nn.Parameter(torch.zeros(()))
        self.loss_time = nn.CrossEntropyLoss()
        self.loss_huber = nn.HuberLoss(reduction='sum', delta=1.0)

    def forward(self, pred, target, pred_arrow, target_time, mask=None):
        num_pixels_mask = torch.sum(mask)
        
        if mask is not None:
            # Fallback de Validação: Se mask.min() == 1.0, não há buracos.
            # Avaliamos a anomalia na imagem inteira sem alterar as predições.
            if mask.min() < 1.0:
                # Composição SOTA: Cola os pixels perfeitos nas áreas visíveis.
                pred = pred * (1.0 - mask) + target * mask
                num_pixels_mask = torch.sum(1.0 - mask)
        
        recon_loss = self.loss_huber(pred, target) / num_pixels_mask
        time_loss = self.loss_time(pred_arrow, target_time.long().squeeze())
        
        if not self.uncertainty_weighting:
            return self.w1 * recon_loss + self.w2 * time_loss

        loss_recon = 0.5 * torch.exp(-self.log_var_recon) * recon_loss + 0.5 * self.log_var_recon
        loss_time = torch.exp(-self.log_var_time) * time_loss + 0.5 * self.log_var_time
        return loss_recon + loss_time
    
def get_criterion(config):
    modelName = config['model']
    if modelName in ST_RECON_TIME_MODEL:
        return ReconArrowLoss(
            uncertainty_weighting=config.get('uncertainty_weighting', True))
    else:
        raise ValueError(f"Invalid model {modelName}")

def create_dali_loaders(config):   
    modelName = config['model']
    dataset = config['dataset']
    print(ST_RECON_TIME_MODEL)

    returnNames = ['sequence', 'masked_sequence', 'mask', 'time_sequence', 'label', 'key', 'prefix']

    if modelName not in ST_RECON_TIME_MODEL:
        raise ValueError(f"Invalid modelName {modelName}")

    if "orig" in dataset:
        pipeline_func = masked_time_pipe
        train_kwargs = dict(num_frames=31, seq_len=9, random_window=True,
                            cover_factor=0.9, square_size=8,
                            augment=config.get('augment', True))
        val_kwargs = dict(num_frames=31, seq_len=9, random_window=False,
                          cover_factor=0.9, square_size=8)
    else:
        raise ValueError(f"Invalid Dataset {dataset}")

    # External-source parallelism: `parallel=True` makes DALI run the Python
    # source in a pool of workers (py_num_workers), which is the main lever for
    # the ~1 batch/s bottleneck. Opt-in via `parallel_source` (default off) because
    # it forks workers and needs per-worker RNG reseeding (see
    # mask_dali_dataloader.seed_worker_if_forked).
    parallel = config.get('parallel_source', False)
    py_num_workers = config.get('py_num_workers', config.get('workers', 4))

    train_pipe = pipeline_func(ann_file=config['train_ann'],
                               batch_size=config['batch_size'],
                               pipe_batch_size=config['batch_size'],
                               num_threads=config['workers'],
                               py_num_workers=py_num_workers,
                               parallel=parallel,
                               train=True, shuffle_temporal=True, mask_prob=0.9,
                               **train_kwargs)
    
    val_pipe = pipeline_func(ann_file=config['val_ann'],
                             batch_size=config['batch_size'],
                             pipe_batch_size=config['batch_size'],
                             num_threads=config['workers'],
                             py_num_workers=py_num_workers,
                             parallel=parallel,
                             train=False, shuffle_temporal=True, mask_prob=0.9,
                             **val_kwargs)
    
    train_pipe.build()
    train_iter = DALIRaggedIterator(train_pipe, returnNames, size=-1, auto_reset=True)
    
    val_pipe.build()
    val_iter = DALIRaggedIterator(val_pipe, returnNames, size=-1, auto_reset=True)

    return train_iter, val_iter

def recon_time_step(model, batch, criterion):
    x = batch[0]['sequence']
    x_masked = batch[0]['masked_sequence']
    x_time = batch[0]['time_sequence']
    l_time = batch[0]['label']
    
    # Slice para pegar apenas 1 canal da máscara: (N, 1, T, H, W)
    mask = batch[0]['mask'][:, 0:1, ...]
    
    y_recon, y_arrow = model(x_masked, x_time)

    loss = criterion(y_recon, x, y_arrow, l_time, mask)
    
    return loss

def get_step_func(config):
    modelName = config['model']
    if modelName in ST_RECON_TIME_MODEL:
        return recon_time_step
    else:
        raise ValueError(f"Invalid model {modelName}")


def train(config):
    device = set_device(use_gpu=config['use_gpu'])

    trainLoader, valLoader = create_dali_loaders(config)

    model = ModelFactory.create_model(model_name=config['model'],
                                      inner_channel=config['in_channel'],
                                      out_channel=config['out_channel'])
    model = model.to(device=device)

    savedModel = config.get('saved_model', None)
    if savedModel:
        state_dict = torch.load(savedModel)
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
        print(f"Loaded saved model {savedModel}")

    criterion = get_criterion(config).to(device)
    # Uncertainty-based weighting adds learnable log-variance params to the
    # criterion; they must be optimised alongside the model.
    optimizer = optim.Adam(lr=config['lr'],
                           params=list(model.parameters()) + list(criterion.parameters()))

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            saveModel(config=config, model=model, epoch=epoch, val_loss=val_loss, val_acc=0.0)

        epochReport = f"Train Loss {train_loss:.4f} Val Loss {val_loss:.4f}\n"
        print(f"Epoch {epoch} LR {optimizer.param_groups[0]['lr']:.4f} {epochReport}")
        print(f"Elapsed Time {time() - start_time:.4f}")

    writer.close()
    print(f"Total time {time() - start_time:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ShanghaiTech TeconTime')

    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()

    config = ConfigParser(args.config_file).config

    train(config)