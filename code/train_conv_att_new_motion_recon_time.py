

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import sys
sys.path.insert(0, "..")

from model_factory import ModelFactory

from mask_dali_dataloader import masked_time_pipe
from motion_mask_dataloader import motion_masked_time_pipe
from nvidia.dali.plugin.pytorch import DALIRaggedIterator

from time import time
from tqdm import tqdm
import argparse

import os

from trainUtils import *
from configParser import ConfigParser

ST_RECON_TIME_MODEL = ['STReconTime','CNN3DReconTimeAttention']

set_seed(12334)


def constant_fill_baseline(target, mask, delta=1.0, fill='mean', eps=1e-8):
    """
    Masked Huber of filling holes with a per-(n,f) statistic of the visible
    voxels, matched to the normalization used by ReconArrowLoss.

    Args:
        target : (N, F, T, H, W) float tensor
        mask   : (N, 1, T, H, W) in {0,1}; 1 = visible, 0 = hole
        delta  : Huber delta (must equal your loss_huber's delta)
        fill   : 'mean' | 'median' | 'zero'

    Returns:
        scalar tensor (same dtype/device as target), or NaN if the batch has
        no holes at all (so callers can skip it in plots).
    """
    assert target.dim() == 5 and mask.dim() == 5, (target.shape, mask.shape)
    assert target.shape[0] == mask.shape[0], (target.shape, mask.shape)
    assert target.shape[2:] == mask.shape[2:], (target.shape, mask.shape)
    assert mask.shape[1] == 1, f"expected single-channel mask, got {mask.shape}"
    assert (mask.min() >= 0) and (mask.max() <= 1), "mask must be in [0,1]"

    N, Fd = target.shape[:2]
    m       = mask.expand_as(target)          # (N,F,T,H,W); broadcast F
    hole    = 1.0 - m
    n_holes = hole.sum()                      # N·T·H·W * (avg hole fraction)
    if n_holes.item() == 0:
        return torch.tensor(float('nan'), device=target.device, dtype=target.dtype)

    if fill == 'mean':
        vis_sum  = (target * m).sum(dim=(2, 3, 4), keepdim=True)   # (N,F,1,1,1)
        vis_cnt  = m.sum(dim=(2, 3, 4), keepdim=True).clamp(min=1.0)
        fill_val = vis_sum / vis_cnt
        filled   = fill_val.expand_as(target)

    elif fill == 'median':
        t  = target.reshape(N, Fd, -1)         # (N,F,T*H*W)
        mm = m.reshape(N, Fd, -1).bool()
        med = torch.zeros(N, Fd, device=target.device, dtype=target.dtype)
        for n in range(N):
            for f in range(Fd):
                v = t[n, f][mm[n, f]]
                if v.numel() > 0:
                    med[n, f] = v.median()
        filled = med.view(N, Fd, 1, 1, 1).expand_as(target)

    elif fill == 'zero':
        filled = torch.zeros_like(target)

    else:
        raise ValueError(fill)

    err = F.huber_loss(filled, target, delta=delta, reduction='none')  # (N,F,T,H,W)
    return (err * hole).sum() / n_holes.clamp(min=1.0)

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

        with torch.no_grad():
            baseline_mean = constant_fill_baseline(target, mask, fill='mean')

        time_loss = self.loss_time(pred_arrow, target_time.long().squeeze())
        
        if not self.uncertainty_weighting:
            return self.w1 * recon_loss + self.w2 * time_loss

        loss_recon = 0.5 * torch.exp(-self.log_var_recon) * recon_loss + 0.5 * self.log_var_recon
        loss_time = torch.exp(-self.log_var_time) * time_loss + 0.5 * self.log_var_time
        total_loss = loss_recon + loss_time
        return total_loss, recon_loss.detach().item(), time_loss.detach().item(), baseline_mean.detach().item()
    
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

    square_size = config.get('square_size', 4)
    cover_factor = config.get('cover_factor', 0.9)

    if "orig" in dataset:
        pipeline_func = masked_time_pipe
        train_kwargs = dict(num_frames=31, seq_len=9, random_window=True,
                            cover_factor=cover_factor, square_size=square_size,
                            augment=config.get('augment', True))
        val_kwargs = dict(num_frames=31, seq_len=9, random_window=False,
                          cover_factor=cover_factor, square_size=square_size)
    elif "full" in dataset:
        pipeline_func = motion_masked_time_pipe
        motion = dict(motion_method=config.get('motion_method', 'consecutive'),
                      motion_aggregate=config.get('motion_aggregate', 'sum'),
                      motion_min_saliency=config.get('motion_min_saliency', 0.01),
                      motion_sigma=config.get('motion_sigma', 1.0),
                      motion_res=config.get('motion_res', 64))
        train_kwargs = dict(seq_len=9, cover_factor=cover_factor, square_size=square_size,
                            augment=config.get('augment', False), **motion)
        val_kwargs = dict(seq_len=9, cover_factor=cover_factor, square_size=square_size, **motion)
    else:
        raise ValueError(f"Invalid Dataset {dataset}")

    train_pipe = pipeline_func(ann_file=config['train_ann'],
                               batch_size=config['batch_size'],
                               pipe_batch_size=config['batch_size'],
                               num_threads=config['workers'],
                               train=True, shuffle_temporal=True, mask_prob=1.0,
                               **train_kwargs)
    
    val_pipe = pipeline_func(ann_file=config['val_ann'],
                             batch_size=config['batch_size'],
                             pipe_batch_size=config['batch_size'],
                             num_threads=config['workers'],
                             train=False, shuffle_temporal=True, mask_prob=1.0,
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
    # optimizer = optim.Adam(lr=config['lr'],
    #                        params=list(model.parameters()) + list(criterion.parameters()))

    optimizer = optim.Adam([{"params": model.parameters(), "lr": config['lr']},
                            {"params": criterion.parameters(), "lr": 1e-4, "weight_decay": 1e-4},])

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
                loss, loss_recon, loss_time, baseline_mean = step_func(model, batch, criterion)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value += loss.detach().item()
                n_batches += 1
                
                desc = f"Epoch {epoch} Loss Recon {loss_recon:.4f} Loss Time {loss_time:.4f} Loss {loss_value/n_batches:.4f}"
                desc = f"{desc} Ratio {loss_recon / baseline_mean:.4f}"
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