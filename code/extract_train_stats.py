import os, sys
import gc
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from models import CNN3DRecon
import base_dataloader as bd
from dataloader_torch import ShanghaiTestSampleDataset
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from configParser import ConfigParser
import argparse

from trainUtils import set_device


os.environ['DALI_DISABLE_NVML'] = '1'

def compute_recon_loss(pred,target,reduction:str='mean'):
    loss = F.l1_loss(pred,target, reduction=reduction) + F.mse_loss(pred,target, reduction=reduction)
    return loss.detach().cpu().item()

def extract_states(config):
    device = set_device(use_gpu=config['use_gpu'])
    
    shape = (config['input_size'][0], config['input_size'][1])
    val_pipe = bd.video_pipe(file_root=config['trainPath'], train=False, shape=shape,
                          sequence_length=10, stride=1, step=10,
                          batch_size=config['batch_size'])
    
    val_pipe.build()
        
        # Criar iterator DALI para PyTorch
    val_loader = DALIGenericIterator([val_pipe],
                                    output_map=['videos', 'labels'],
                                    auto_reset=True,
                                    reader_name='seq')
    
    checkpoint = config['saved_model']
    print(f"Loading {checkpoint}")
    reconModel = CNN3DRecon(in_channel=config['in_channel'], out_channel=config['out_channel'])
    state_dict = torch.load(checkpoint)
    reconModel.load_state_dict(state_dict['model_state_dict'], strict=True)
    reconModel = reconModel.to(device)
    

    err_values = []

    with torch.no_grad():
        reconModel.eval()
        vloader = tqdm(val_loader, unit='batch')
        tic = time()
        
        for i,batch in enumerate(vloader):
            clip = batch[0]['videos']
            pred = reconModel(clip)
            pred = pred.sigmoid()

            num_batches = clip.shape[0]
            num_frames = clip.shape[2]

            recon_err = []
            for b in range(num_batches):
                err = []
                for f in range(num_frames):
                    diff = compute_recon_loss(pred[b,:,f,:,:], clip[b,:,f,:,:])
                    err.append(diff)
                err = np.array(err)
                recon_err.extend(err)

            recon_err = np.array(recon_err)
            err_values.extend(recon_err)

            desc = f"Batch {i} - Shape {clip.shape} Elapsed Time {time()-tic:.3f} "
            vloader.set_description(desc)
    
    err_values = np.array(err_values)
    print(f"Saving stats")
    np.save(f"stats_train_{os.path.basename(checkpoint.replace(".pth",""))}.npy",err_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ShanghaiTech CamClassifier Self-Sup')

    parser.add_argument('--config_file', type=str, required=True, help='Seu nome')
    args = parser.parse_args()

    config = ConfigParser(args.config_file).config
    extract_states(config)