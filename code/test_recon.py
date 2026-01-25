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

from trainUtils import set_device
from model_factory import ModelFactory

os.environ['DALI_DISABLE_NVML'] = '1'
__allow_list__ = (".jpg", ".jpeg", ".png", ".tiff", ".tif")

def extract_metrics(result_raw_dict_path:str, sigma:float=3.5):
    from torch.distributions import Normal
    import math
    from sklearn.metrics import roc_auc_score, roc_curve

    result_raw_dict = np.load(result_raw_dict_path, allow_pickle=True)

    def gaussian_kernel_1d(sigma: float = 3.5, num_sigmas: float = 3.) -> torch.Tensor:
        radius = math.ceil(num_sigmas * sigma)
        support = torch.arange(-radius, radius + 1, dtype=torch.float)
        kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()
        return kernel.mul_(1 / kernel.sum())
    
    auc_arr = {}
    with torch.no_grad():
        gaussian_filter = gaussian_kernel_1d(sigma=sigma)

        for key in result_raw_dict.keys():
            raw_err = result_raw_dict[key][()]['recon_err']
            label = result_raw_dict[key][()]['labels']

            raw_err_torch = torch.from_numpy(raw_err).reshape(1,-1)
            pad_head = raw_err[0]*torch.ones(gaussian_filter.shape[0]//2).reshape(1,-1)
            pad_tail = raw_err[-1]*torch.ones(gaussian_filter.shape[0]//2).reshape(1,-1)

            raw_err_pad = torch.cat((pad_head,raw_err_torch,pad_tail), dim=1).to(torch.float)

            smooth_err = F.conv1d(raw_err_pad, weight=gaussian_filter.view(1, 1, -1), padding='valid').squeeze()

            try:
                macro_auc = roc_auc_score(label, smooth_err)
                
                if not math.isnan(macro_auc):
                    auc_arr[key] = macro_auc
            except:
                print(f"Error for {key}")
    

    dest_name = os.path.basename(result_raw_dict_path).replace("results_raw_","").replace(".npz","")
    np.savez(f"macro_auc_{dest_name}.npz",**auc_arr)
    result_raw_dict.close()

    return auc_arr

def compute_recon_score(pred, target):
    """pred and target shape: (B,C,F,H,W) """
    
    pred = pred.permute(0,2,3,4,1) # (B,C,F,H,W) -> (B,F,H,W,C)
    target = target.permute(0,2,3,4,1) # (B,C,F,H,W) -> (B,F,H,W,C)
    diff = torch.abs(pred - target)
    mean = diff.mean(dim=(2,3,4)) # (B,F,H,W,C) -> (B,F)

    return mean.detach().cpu()

def compute_recon_loss(pred,target,reduction:str='mean'):
    loss = F.l1_loss(pred, target, reduction=reduction) + F.mse_loss(pred, target, reduction=reduction)
    return loss.detach().cpu()

def eval_sample(model, video_path, label_path, config, transform, device='cpu', workers=4):
    print(f"Evaluating video {video_path}")
    
    dataset = ShanghaiTestSampleDataset(video_path=video_path,
                                label_path=label_path,
                                transform=transform,
                                encoding=True)

    loader = DataLoader(dataset, batch_size = len(dataset),
                        shuffle=False, num_workers = workers,
                        pin_memory=True)
    
    it = iter(loader)
    with torch.no_grad():
        clip, label = next(it)
        clip = clip.to(device)
        predClip = model(clip)

        print(f"Clip {clip.shape}")

        recon_scores = compute_recon_score(predClip, clip)
        print(recon_scores.shape)
        recon_err = recon_scores.ravel().numpy()
    
    del clip
    del predClip
    del it
    del loader
    del dataset
    torch.cuda.empty_cache()
    gc.collect()

    return recon_err, label.numpy().ravel()

def load_clip(clip:list, transf, device='cpu'):
    frames = []
    for file in clip:
        frame = cv2.imread(file, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).to(device=device)
        frame = frame.permute(2,0,1) #C,H,W
        frame = transf(frame)
        frames.append(frame)
    frames = torch.stack(frames, dim=0) #F,C,H,W
    frames = frames.permute(1,0,2,3) #C,F,H,W
    frames = frames.unsqueeze(dim=0) #1,C,F,H,W

    return frames

def eval_sample_detection(model, video_path, label_path, config, transform, device='cpu'):
    print(f"Evaluating video {video_path}")
    seqLen = config['seqLen']
    fstride = config['fstride']
    cstride = config['cstride']
    
    labels = np.load(label_path)
    frame_dirs = [os.path.join(video_path, frame_dir) for frame_dir in os.listdir(video_path)]
    
    assert len(frame_dirs) == labels.shape[0]
    
    scores = []
    for i, frame_dir in enumerate(frame_dirs):
        frames = [os.path.join(frame_dir, file) for file in os.listdir(frame_dir) if file.endswith(__allow_list__)]
        
        recon_err = 0.0
        if len(frames) != 0:
            with torch.no_grad():
                for start in range(0, len(frames) - (seqLen * fstride) + 1, cstride):
                    end = start + seqLen * fstride
                    clip_files = frames[start:end:fstride] #(F,)
                    clip = load_clip(clip_files, transform, device=device) #(1,C,F,H,W)
                    predClip = model(clip)

                    recon_scores = compute_recon_score(predClip, clip) # (1,F)
                    recon_err += recon_scores.ravel().mean().numpy()
        
        scores.append(recon_err)
    
    assert len(scores) == labels.shape[0]
    
    return np.array(scores), labels

def load_model(config):
    reconModel = ModelFactory.create_model(model_name=config['model'],
                                           in_channel=config['in_channel'],
                                           out_channel=config['out_channel'])
    
    print(f"Loading {checkpoint}")
    state_dict = torch.load(checkpoint)
    reconModel.load_state_dict(state_dict['model_state_dict'], strict=True)
    reconModel = reconModel.to(device)
    reconModel.eval()

    return reconModel

if __name__ == "__main__":
    configFile = "configReconObj.json"
    config = ConfigParser(configFile).config
    checkpoint = config['saved_model']
    testConfig = config.get('testPath',{})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_func = eval_sample_detection if "Obj" in configFile else eval_sample

    shape = (config['input_size'][0], config['input_size'][1])
    transf = v2.Compose([v2.Resize(size=shape, antialias=True),
                        v2.ToDtype(torch.float32, scale=True),])

    sample_ids = [os.path.basename(folder) for folder in os.listdir(config['testPath']['clips'])]

    recon_model = load_model(config)
    
    result_dir = {}
    for sample_id in sample_ids:
        print(f"Evaluating {sample_id}")
        sample_path = os.path.join(testConfig['clips'], sample_id)
        label_path = os.path.join(testConfig['labels'],f"{sample_id}.npy")
        recon_err, labels = eval_func(recon_model,
                                     video_path=sample_path, label_path=label_path,
                                     config=config,
                                     transform=transf, device=device)
        result_dir[sample_id] = {"recon_err": recon_err, "labels": labels}

    del recon_model
    torch.cuda.empty_cache()
    gc.collect()
    
    model_name = os.path.basename(checkpoint).replace('.pth','')
    dest_folder = "./results"
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    dest_file = os.path.join(dest_folder,f"results_raw_{model_name}.npz")
    
    print(f"Saving results from model {model_name} to {dest_file}")
    np.savez(dest_file,**result_dir)

    macro_auc_values = extract_metrics(dest_file)

    mean_auc = 0.0
    for key in macro_auc_values.keys():
        macro_auc = macro_auc_values[key]
        mean_auc += macro_auc

    mean_auc = mean_auc/len(macro_auc_values.keys())
    print(f"Mean AUC {mean_auc}")

        


