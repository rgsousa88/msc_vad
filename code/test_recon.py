import os, sys
import gc
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from models import CNN3DRecon
from dataloader_torch import ShanghaiTestSampleDataset

from configParser import ConfigParser

os.environ['DALI_DISABLE_NVML'] = '1'

def compute_recon_loss(pred,target,reduction:str='mean'):
    loss = F.l1_loss(pred,target, reduction=reduction) + F.mse_loss(pred,target, reduction=reduction)
    return loss.detach().cpu().item()

def eval_sample(model, sample_id, config, transform, device='cpu'):
    testConfig = config.get('testPath',{})
    workers = config.get('workers', 4)

    video_path = os.path.join(testConfig['clips'], sample_id)
    label_path = os.path.join(testConfig['labels'],f"{sample_id}.npy")

    dataset = ShanghaiTestSampleDataset(video_path=video_path,
                                label_path=label_path,
                                transform=transform,
                                encoding=True)

    loader = DataLoader(dataset, batch_size = len(dataset),
                        shuffle=False, num_workers = workers,
                        pin_memory=True)

    with torch.no_grad():
        model.eval()
        it = iter(loader)
        clip, label = next(it)
        clip = clip.to(device)
        predClip = model(clip)
        predClip = predClip.sigmoid()

        num_batches = clip.shape[0]
        num_frames = clip.shape[2]

        recon_err = []
        for b in range(num_batches):
            err = []
            for f in range(num_frames):
                diff = compute_recon_loss(predClip[b,:,f,:,:], clip[b,:,f,:,:])
                err.append(diff)
            err = np.array(err)
            recon_err.extend(err)

        recon_err = np.array(recon_err)
    
    del clip
    del predClip
    del dataset
    del loader
    torch.cuda.empty_cache()
    gc.collect()

    return recon_err, label.numpy().ravel()


if __name__ == "__main__":
    configFile = "configRecon.json"
    config = ConfigParser(configFile).config
    checkpoint = config['saved_model'] #"/home/rgadelha/msc_vad/code/checkpoint/cnn3d_recon_in32out64_20251211_112044.pth"
    shape = (config['input_size'][0], config['input_size'][1])
    transf = v2.Compose([v2.Resize(size=shape, antialias=True),
                        v2.ToDtype(torch.float32, scale=True),])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {checkpoint}")
    reconModel = CNN3DRecon(in_channel=config['in_channel'], out_channel=config['out_channel'])
    state_dict = torch.load(checkpoint)
    reconModel.load_state_dict(state_dict['model_state_dict'], strict=True)
    reconModel = reconModel.to(device)
    reconModel.eval()

    sample_ids = [os.path.basename(folder) for folder in os.listdir(config['testPath']['clips'])]
    
    result_dir = {}
    for sample_id in sample_ids:
        print(f"Evaluating {sample_id}")
        recon_err, labels = eval_sample(reconModel, sample_id, config, transf, device=device)
        result_dir[sample_id] = {"recon_err": recon_err, "labels": labels}

    del reconModel
    del state_dict
    torch.cuda.empty_cache()
    gc.collect()
    
    model_name = os.path.basename(checkpoint)
    dest_folder = "./results"
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    dest_file = os.path.join(dest_folder,f"{model_name.replace('.pth','')}.npz")
    print(f"Saving results from model {model_name} to {dest_file}")
    np.savez(dest_file,**result_dir)

        


