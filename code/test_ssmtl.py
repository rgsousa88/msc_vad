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

from models import SSMTLModel
from dataloader_torch import SSMTLModelDataset

from configParser import ConfigParser

from trainUtils import set_device

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

def compute_anomaly_scores(model, annotation_path:str, config, workers:int = 4, device="cpu"):
    print(f"Evaluating annotation {annotation_path}")

    testDataset = SSMTLModelDataset(annotation_path=annotation_path,
                                    input_size=config['input_size'],
                                    window=3,
                                    is_test=True)
    
    testLoader = DataLoader(testDataset, batch_size=config['batch_size'], shuffle=False, num_workers=workers)
    
    result_scores_dict = {}

    with torch.no_grad():
        for idx, batch in enumerate(testLoader):
            (x_arrow, l_arrow), (x_motion, l_motion), x_recon, (x_distil, feat_distil), key = batch
            
            x_arrow = x_arrow.to(device)
            l_arrow = l_arrow.to(device)
            x_motion = x_motion.to(device)
            l_motion = l_motion.to(device)
            x_recon = x_recon.to(device)
            x_distil = x_distil.to(device)
            feat_distil = feat_distil.to(device)

            y_arrow, y_motion, y_recon, y_distil = model(x_arrow, x_motion, x_recon, x_distil)

            score_arrow = F.softmax(y_arrow,dim=1)
            score_motion = F.softmax(y_motion,dim=1)
            score_distill = torch.abs(y_distil[:,1000:] - feat_distil[:,1000:]).mean(dim=1)
            score_recon = torch.abs(y_recon - x_distil.reshape(y_recon.shape)).mean(dim=(1,2,3))

            # print(f"Score arrow {score_arrow[:,1].detach().cpu().numpy()}")
            # print(f"Score motion {score_motion[:,1].detach().cpu().numpy()}")
            # print(f"Score distil {score_distill.detach().cpu().numpy()}")
            # print(f"Score recon {score_recon.detach().cpu().numpy()}")

            score = 0.25 * (score_arrow[:,1] + score_motion[:,1] + score_distill + score_recon)

            for i,k in enumerate(key):
                if not k in result_scores_dict.keys():
                    result_scores_dict[k] = []
                result_scores_dict[k].append(score[i].detach().cpu().numpy())
    
            del x_arrow, l_arrow, x_motion, l_motion, x_recon, x_distil, feat_distil
            del y_arrow, y_motion, y_recon, y_distil
            del score, score_arrow, score_motion, score_distill, score_recon

    del testLoader
    del testDataset
    torch.cuda.empty_cache()
    gc.collect()

    return result_scores_dict

def load_model(config, device="cpu"):
    model = SSMTLModel(in_channel=config['in_channel'],
                       out_channel=config['out_channel'])
    
    print(f"Loading {config['saved_model']}")
    state_dict = torch.load(config['saved_model'])
    model.load_state_dict(state_dict['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()

    return model

if __name__ == "__main__":
    configFile = "configSSMTL.json"
    config = ConfigParser(configFile).config
    checkpoint = config['saved_model']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_path = config['test_ann_path']

    annotation_files = [os.path.join(test_path, file) for file in os.listdir(test_path) if file.endswith(".csv")]

    model = load_model(config, device=device)
    
    result_dir = {}
    for ann_file in annotation_files:
        print(f"Evaluating {ann_file}")
        sample_id = os.path.basename(ann_file).replace("annotation_",'').replace(".csv",'')
        label_path = os.path.join(config['test_label_path'],f"{sample_id}.npy")
        labels = np.load(label_path)
        result_scores_dict = compute_anomaly_scores(model, ann_file, config, device=device)
        result_dir[sample_id] = {"scores_per_frame": result_scores_dict, "labels": labels}
        break

    del model
    torch.cuda.empty_cache()
    gc.collect()

    for key in result_dir:
        print(f"Sample {key}")
        for frame_key in result_dir[key]["scores_per_frame"]:
            scores = result_dir[key]["scores_per_frame"][frame_key]
            print(f"Frame {frame_key} Score {max(scores)}")
        print(f"Labels {result_dir[key]["labels"]}")
    
    # model_name = os.path.basename(checkpoint).replace('.pth','')
    # dest_folder = "./results"
    # if not os.path.exists(dest_folder):
    #     os.makedirs(dest_folder)
    # dest_file = os.path.join(dest_folder,f"results_raw_{model_name}.npz")
    
    # print(f"Saving results from model {model_name} to {dest_file}")
    # np.savez(dest_file,**result_dir)

    # macro_auc_values = extract_metrics(dest_file)

    # mean_auc = 0.0
    # for key in macro_auc_values.keys():
    #     macro_auc = macro_auc_values[key]
    #     mean_auc += macro_auc

    # mean_auc = mean_auc/len(macro_auc_values.keys())
    # print(f"Mean AUC {mean_auc}")

        


