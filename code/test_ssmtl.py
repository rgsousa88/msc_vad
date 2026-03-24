import os, sys

os.environ['DALI_DISABLE_NVML'] = '1'

import warnings
warnings.filterwarnings('ignore')

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

from model_factory import ModelFactory
from dataloader_torch import SSMTLModelDataset

from dali_dataloader import ssmtl_pipe, masked_pipe, masked_arrow_pipe
from nvidia.dali.plugin.pytorch import DALIRaggedIterator

from configParser import ConfigParser

from trainUtils import set_device

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
            raw_err = result_raw_dict[key][()]['scores']
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

def decode_key(key, prefix):
    dec = bytes(key)
    dec_prefix = bytes(prefix)
    dec = dec.removeprefix(dec_prefix)
    dec = dec.decode('utf-8')
    return dec

def build_loader(config:dict, annotation_path:str, workers:int=4, device="cpu"):
    modelName = config['model']
    if modelName == 'SSMTLModel':
        returnNames = ['x_arrow', 'l_arrow', 'x_motion', 'l_motion', 'x_recon', 'x_distil', 'feat_distil','key','prefix']
        test_pipe = ssmtl_pipe(ann_file=annotation_path,
                            num_frames=7,
                            batch_size=config['batch_size'],
                            num_threads=config['workers'],
                            train=False)
    elif modelName == 'SSMTLAutoencoder':
        returnNames = ['sequence', 'masked_sequence', 'key', 'prefix']
        test_pipe = masked_pipe(ann_file=annotation_path,
                                num_frames=7,
                                batch_size=config['batch_size'],
                                num_threads=config['workers'],
                                train=False)
    elif modelName == 'SSTMLAutoEncArrow':
        returnNames = ['sequence', 'masked_sequence', 'seq_backward', 'label_backward','key', 'prefix']
        test_pipe = masked_arrow_pipe(ann_file=annotation_path,
                                num_frames=7,
                                batch_size=config['batch_size'],
                                num_threads=config['workers'],
                                train=False)
    else:
        raise ValueError(f"Invalid modelName {modelName}")
    
    test_pipe.build()
    testLoader = DALIRaggedIterator(test_pipe, returnNames, size=-1)

    return testLoader

def compute_ssmtl_score(model, batch):
    x_arrow, x_motion = batch[0]['x_arrow'], batch[0]['x_motion']
    x_recon, x_distil = batch[0]['x_recon'], batch[0]['x_distil']
    feat_distil = batch[0]['feat_distil']

    y_arrow, y_motion, y_recon, y_distil = model(x_arrow, x_motion, x_recon, x_distil)

    score_arrow = F.softmax(y_arrow, dim=1)
    score_motion = F.softmax(y_motion, dim=1)
    score_distill = torch.abs(y_distil[:,1000:] - feat_distil[:,1000:]).mean(dim=1)
    score_recon = torch.abs(y_recon - x_distil.reshape(y_recon.shape)).mean(dim=(1,2,3))

    score = 0.25 * (score_arrow[:,1] + score_motion[:,1] + score_distill + score_recon)

    return score

def compute_ssmtl_recon_score(model, batch):
    x = batch[0]['sequence']
    y_recon = model(x)
    
    score = torch.abs(y_recon - x.reshape(y_recon.shape)).mean(dim=(1,3,4))
    return score

def compute_ssmtl_recon_arrow_score(model, batch):
    x, x_arrow = batch[0]['sequence'], batch[0]['seq_backward']
    y_recon, y_arrow = model(x, x_arrow)

    score_arrow = F.softmax(y_arrow, dim=1)
    score_recon = torch.abs(y_recon - x.reshape(y_recon.shape)).mean(dim=(1,3,4))

    score = 0.5 * (score_arrow[:,1] + torch.max(score_recon, dim=1).values)
    
    return score

def build_score_func(config:dict):
    modelName = config['model']
    if modelName == 'SSMTLModel':
        return compute_ssmtl_score
    elif modelName == 'SSMTLAutoencoder':
        return compute_ssmtl_recon_score
    elif modelName == 'SSTMLAutoEncArrow':
        return compute_ssmtl_recon_arrow_score
    else:
        raise ValueError(f"Invalid modelName {modelName}")

def compute_anomaly_scores(model, annotation_path:str, config, workers:int = 4, device="cpu"):
    print(f"Evaluating annotation {annotation_path}")

    testLoader = build_loader(config=config, annotation_path=annotation_path, workers=workers, device=device)
    score_func = build_score_func(config=config)
    
    result_scores_dict = {}

    with torch.no_grad():
        for idx, batch in enumerate(testLoader):
            score = score_func(model, batch)
            score = score.detach().cpu().numpy()

            key, prefix = batch[0]['key'], batch[0]['prefix']
            key = key.detach().cpu().numpy()
            prefix = prefix.detach().cpu().numpy()

            for i,k in enumerate(key):
                deckey = decode_key(k, prefix[i])
                if not deckey in result_scores_dict.keys():
                    result_scores_dict[deckey] = []
                result_scores_dict[deckey].append(score[i])

    del testLoader
    torch.cuda.empty_cache()

    return result_scores_dict

def load_model(config, device="cpu"):
    factory = ModelFactory()

    model = factory.create_model(model_name=config['model'],
                                 in_channel=config['in_channel'],
                                 out_channel=config['out_channel'])
    
    print(f"Loading {config['saved_model']}")
    checkpoint = torch.load(config['saved_model'])

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Create a new state dict without the prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()

    config = ConfigParser(args.config_file).config

    checkpoint = config['saved_model']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_path = config['test_ann_path']

    annotation_files = [os.path.join(test_path, file) for file in os.listdir(test_path) if file.endswith(".csv")]

    model = load_model(config, device=device)
    
    result_dict = {}
    for ann_file in annotation_files:
        print(f"Evaluating {ann_file}")
        sample_id = os.path.basename(ann_file).replace("annotation_",'').replace(".csv",'')
        label_path = os.path.join(config['test_label_path'],f"{sample_id}.npy")
        labels = np.load(label_path)
        result_scores_dict = compute_anomaly_scores(model, ann_file, config, device=device)

        scores = []
        frame_idx = []
        for frame_key in sorted(result_scores_dict.keys()):
            frame_scores = result_scores_dict[frame_key]
            idx = int(frame_key.split('_')[-1])
            score = np.max(frame_scores).astype('float32')
            frame_idx.append(idx)
            scores.append(score)
        scores = np.array(scores)
        result_dict[sample_id] = {"scores":scores, "labels":labels[frame_idx]}

    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    model_name = os.path.basename(checkpoint).replace('.pth','')
    dest_folder = "./results"
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    dest_file = os.path.join(dest_folder,f"results_raw_{model_name}.npz")
    
    print(f"Saving results from model {model_name} to {dest_file}")
    np.savez(dest_file,**result_dict)

    macro_auc_values = extract_metrics(dest_file)

    mean_auc = 0.0
    for key in macro_auc_values.keys():
        macro_auc = macro_auc_values[key]
        mean_auc += macro_auc

    mean_auc = mean_auc/len(macro_auc_values.keys())
    print(f"Mean AUC {mean_auc}")

        


