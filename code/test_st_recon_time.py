import os, sys

import warnings
warnings.filterwarnings('ignore')

import gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_factory import ModelFactory

from mask_dali_dataloader import masked_time_pipe
from nvidia.dali.plugin.pytorch import DALIRaggedIterator

from configParser import ConfigParser

from trainUtils import set_device

__allow_list__ = (".jpg", ".jpeg", ".png", ".tiff", ".tif")

from train_st_recon_time import ST_RECON_TIME_MODEL

from torch.distributions import Normal
import math
from sklearn.metrics import roc_auc_score, roc_curve, auc

def gaussian_filter_(support, sigma):
    mu = support[len(support) // 2 - 1]
    filter_values = 1.0 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter_values.astype('float32')

def gaussian_kernel_1d(sigma: float = 3.5, num_sigmas: float = 3.) -> torch.Tensor:
    radius = math.ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()
    return kernel.mul_(1 / kernel.sum())

def normalize(err_):
    err_ = np.array(err_, dtype=np.float32)
    if len(err_) == 0:
        return err_
    
    err_min = np.min(err_)
    err_max = np.max(err_)
    
    # Avoid division by zero
    if err_max - err_min > 1e-10:
        err_ = err_ - err_min
        err_ = err_ / (err_max - err_min)
    else:
        err_ = np.zeros_like(err_)
    
    return err_

def extract_metrics(result_raw_dict:str | dict, sigma:float=3.5, padding:str='orig'):
    dest_path = None

    if isinstance(result_raw_dict, str):
        print(f"Loading {result_raw_dict}")
        dest_name = os.path.basename(result_raw_dict).replace("results_raw_","").replace(".npz","")
        dest_path = os.path.join("auc_files", f"macro_auc_{dest_name}_{padding}_v2.npz")
        
        with np.load(result_raw_dict, allow_pickle=True) as npz:
            result_raw_dict = {k: npz[k].item() for k in npz.files}
    
    auc_arr = {}
    with torch.no_grad():
        gaussian_filter = gaussian_filter_(np.arange(1, 202), 31)

        for key in result_raw_dict.keys():
            raw_err = result_raw_dict[key]['scores'] #result_raw_dict[key][()]['scores']
            label = result_raw_dict[key]['labels'] #result_raw_dict[key][()]['labels']

            padding_size = len(gaussian_filter) // 2

            if padding == 'orig':
            # Pad the signal for correlation
                in_padded = np.concatenate((
                    raw_err[:padding_size],  # Mirror padding at start
                    raw_err,
                    raw_err[-padding_size:]  # Mirror padding at end
                ))

            elif padding == 'mirror':
            # Fixed padding concept
                in_padded = np.concatenate((
                                raw_err[-padding_size:],  # Mirror padding at start
                                raw_err,
                                raw_err[:padding_size]  # Mirror padding at end
                            ))
            elif padding == 'copy':
            # Copied padding 
                in_padded = np.concatenate((
                                raw_err[0]*np.ones((gaussian_filter.shape[0]//2,)).reshape(-1,),  # Mirror padding at start
                                raw_err, 
                                raw_err[-1]*np.ones((gaussian_filter.shape[0]//2,)).reshape(-1,)  # Mirror padding at end
                            ))

            else:
                raise ValueError(f"Invalid padding {padding}")

            smooth_err = np.correlate(in_padded, gaussian_filter, 'valid')

            try:
                #macro_auc = roc_auc_score(label, smooth_err)

                if padding == 'orig':
                    fpr, tpr, _ = roc_curve(np.concatenate(([0], label, [1])), 
                                            np.concatenate(([0], smooth_err, [1])))

                else:
                    fpr, tpr, _ = roc_curve(label, smooth_err)
                
                macro_auc = auc(fpr, tpr)

                if not math.isnan(macro_auc):
                    auc_arr[key] = macro_auc
            except:
                print(f"Error for {key}")

    if dest_path:
        np.savez(dest_path, **auc_arr)

    return auc_arr

def decode_key(key, prefix):
    dec = bytes(key)
    dec_prefix = bytes(prefix)
    dec = dec.removeprefix(dec_prefix)
    dec = dec.decode('utf-8')
    return dec

def build_loader(config:dict, annotation_path:str, workers:int=4, device="cpu"):
    modelName = config['model']
    dataset = config['dataset']

    if modelName in ST_RECON_TIME_MODEL:
        returnNames = ['sequence', 'masked_sequence', 'mask', 'time_sequence', 'label', 'key', 'prefix']
        # The reconstruction branch is trained as an inpainter (tube holes in
        # ~90% of train samples), so evaluate it under the same regime: apply
        # the mask at test too. `test_mask_prob=1.0` guarantees every clip has
        # holes; use the same cover_factor/square_size as create_dali_loaders
        # in train_st_recon_time.py so train and test mask statistics match.
        test_mask_prob = config.get('test_mask_prob', 1.0)
        if "orig" in dataset:
            test_pipe = masked_time_pipe(ann_file=annotation_path,
                                    num_frames=config['seqLen'],
                                    seq_len=config['seqLen'],
                                    batch_size=config['batch_size'],
                                    pipe_batch_size=config['batch_size'],
                                    num_threads=config['workers'],
                                    train=False,
                                    shuffle_temporal=False,
                                    mask_prob=test_mask_prob,
                                    cover_factor=0.9, square_size=8,
                                    random_window=False)
        else:
            raise ValueError(f"Invalid Dataset {dataset}")
    else:
        raise ValueError(f"Invalid modelName {modelName}")
    
    test_pipe.build()
    testLoader = DALIRaggedIterator(test_pipe, returnNames, size=-1, auto_reset=True)

    return testLoader

def compute_recon_time_score(model, batch):
    x = batch[0]['sequence']
    x_time = batch[0]['time_sequence']

    x_masked = batch[0]['masked_sequence']# entrada com máscara
    mask = batch[0]['mask'][:, 0:1, ...]

    # y_recon, y_time = model(x, x_time)
    y_recon, y_time = model(x_masked, x_time)

    # Máscara de buracos: 1 onde é buraco, 0 onde é visível
    hole_mask = 1.0 - mask
    num_holes = hole_mask.sum(dim=(1,2,3,4))  # soma por amostra do batch

    if (num_holes == 0).any():
        recon_error_all = torch.abs(y_recon - x.reshape(y_recon.shape))
        score_rec_no_holes = recon_error_all.mean(dim=(1,2,3,4))
    else:
        score_rec_no_holes = None

    recon_error_masked = torch.abs(y_recon - x.reshape(y_recon.shape)) * hole_mask
    score_rec_holes = recon_error_masked.sum(dim=(1,2,3,4)) / num_holes.clamp(min=1e-8)

    score = torch.where(num_holes > 0, score_rec_holes, 
                           score_rec_no_holes if score_rec_no_holes is not None else score_rec_holes)
    
    score_time = 1.0 - F.softmax(y_time, dim=1)[:,0]
    #print(f"Score {score.mean().detach().cpu().item()} Score Time {score_time.mean().detach().cpu().item()}")
    return score + score_time

def compute_recon_time_score_v2(model, batch):
    x = batch[0]['sequence']
    x_time = batch[0]['time_sequence']

    x_masked = batch[0]['masked_sequence']# entrada com máscara
    # mask: 1 = visible, 0 = hole (shared tube holes across the clip)
    mask = batch[0]['mask'][:, 0:1, ...]

    # Feed the masked input, matching the training-time inpainting objective
    # (the previous version fed the clean sequence and scored the whole frame,
    # which mismatched how the reconstruction branch was trained).
    y_recon, y_time = model(x_masked, x_time)

    err = torch.abs(y_recon - x.reshape(y_recon.shape))   # (N, C, T, H, W)
    hole = 1.0 - mask                                     # (N, 1, T, H, W)

    # Per-frame mean error over the invented (hole) pixels ...
    num_holes = hole.sum(dim=(1, 3, 4))                   # (N, T)
    hole_err = (err * hole).sum(dim=(1, 3, 4)) / num_holes.clamp(min=1e-8)
    # ... falling back to the whole frame when a sample has no holes.
    full_err = err.mean(dim=(1, 3, 4))                    # (N, T)
    per_frame = torch.where(num_holes > 0, hole_err, full_err)

    score = per_frame.amax(dim=1)                         # (N,)

    score_time = 1.0 - F.softmax(y_time, dim=1)[:,0]
    #print(f"Score T {F.softmax(y_time, dim=1).detach().cpu()}")
    #print(f"Score {score.mean().detach().cpu().item()} Score Time {score_time.mean().detach().cpu().item()}")
    return score, score_time

def build_score_func(config:dict):
    modelName = config['model']
    if modelName in ST_RECON_TIME_MODEL:
        return compute_recon_time_score_v2
    else:
        raise ValueError(f"Invalid modelName {modelName}")

def compute_anomaly_scores(model, annotation_path:str, config, workers:int = 4, device="cpu"):
    print(f"Evaluating annotation {annotation_path}")

    testLoader = build_loader(config=config, annotation_path=annotation_path, workers=workers, device=device)
    score_func = build_score_func(config=config)
    
    result_scores_dict = {}

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(testLoader):
            score = score_func(model, batch)
            key, prefix = batch[0]['key'], batch[0]['prefix']
            key = key.detach().cpu().numpy()
            prefix = prefix.detach().cpu().numpy()

            if torch.is_tensor(score):
                score = score.detach().cpu().numpy()
    
                for i,k in enumerate(key):
                    deckey = decode_key(k, prefix[i])
                    if not deckey in result_scores_dict.keys():
                        result_scores_dict[deckey] = []
                    #print(f"res[{deckey}] = {score[i]}")
                    result_scores_dict[deckey].append(score[i])
            else:
                # print(f"Score Tuple {score}")
                score_r, score_t = score
                score_r = score_r.detach().cpu().numpy()
                score_t = score_t.detach().cpu().numpy()

                for i,k in enumerate(key):
                    deckey = decode_key(k, prefix[i])
                    if not deckey in result_scores_dict.keys():
                        result_scores_dict[deckey] = []
                    # print(f"res[{deckey}] = {score_r[i]},{score_t[i]}")
                    result_scores_dict[deckey].append((score_r[i], score_t[i]))            

    del testLoader
    #exit()

    return result_scores_dict

def load_model(config, device="cpu"):
    factory = ModelFactory()

    model = factory.create_model(model_name=config['model'],
                                 inner_channel=config['in_channel'],
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

def ppt_plot(scores, labels):
    import plotext as ptt

    fig = ptt.figure
    fig.clear()

    sig1 = fig.signal(labels).lines()
    sig2 = fig.signal(scores).lines()

    fig.draw(sig1)
    fig.draw(sig2)
    fig.show()

def compute_anomaly_score_per_video(model, config, ann_file:str):
    sample_id = os.path.basename(ann_file).replace("annotation_",'').replace(".csv",'')
    label_path = os.path.join(config['test_label_path'],f"{sample_id}.npy")
    labels = np.load(label_path)
    result_scores_dict = compute_anomaly_scores(model, ann_file, config, device=device)

    scores = [0.0] * len(labels)
    frame_idx = []

    # Update according logic of score v2
    for frame_key in sorted(result_scores_dict.keys()):
        #print(f"Frame {frame_key}")
        ocs_frame_scores = result_scores_dict[frame_key]
        #print(f"\t\tOCS Frame Scores {len(ocs_frame_scores)}")
        idx = int(frame_key.split('_')[-1])
        mean_ocs_errs = []
        for i, ocs in enumerate(ocs_frame_scores):
            mean_ocs_errs.append(np.mean(ocs))
        #print(f"\t\tLen Mean OCS {len(mean_ocs_errs)}")
        scores[idx] = np.max(mean_ocs_errs).astype('float32')
        frame_idx.append(idx)
    
    scores = np.array(scores)
    scores = normalize(scores)
    
    #print(f"Scores {scores} Labels {labels}")
    key, value = sample_id, {"scores":scores, "labels":labels}
    #exit()

    return key, 

def compute_anomaly_score_per_video_v2(model, config, ann_file:str):
    sample_id = os.path.basename(ann_file).replace("annotation_",'').replace(".csv",'')
    label_path = os.path.join(config['test_label_path'],f"{sample_id}.npy")
    labels = np.load(label_path)
    result_scores_dict = compute_anomaly_scores(model, ann_file, config, device=device)

    scores_r = [0.0] * len(labels)
    scores_t = [0.0] * len(labels)
    frame_idx = [0.0] * len(labels)

    # Update according logic of score v2
    for frame_key in sorted(result_scores_dict.keys()):
        # print(f"Frame {frame_key}")
        ocs_frame_scores = result_scores_dict[frame_key]
        #print(f"\t\tOCS Frame Scores {len(ocs_frame_scores)}")
        idx = int(frame_key.split('_')[-1])
        ocs_errs_r = []
        ocs_errs_t = []
        for i, ocs_tuple in enumerate(ocs_frame_scores):
            ocs_frame_score_r, ocs_frame_score_t = ocs_tuple
            ocs_errs_r.append(ocs_frame_score_r)
            ocs_errs_t.append(ocs_frame_score_t)
        # print(f"\t\tOCS Recon {ocs_errs_r}")
        # print(f"\t\tOCS Time {ocs_errs_t}")
        scores_r[idx] = np.max(np.array(ocs_errs_r)).astype('float32')
        scores_t[idx] = np.max(np.array(ocs_errs_t)).astype('float32')
        frame_idx[idx] = frame_key

    # for i,score_tuple in enumerate(zip(scores_r, scores_t)):
    #     r,t = score_tuple
    #     print(f"Frame {i} Scores R {r} Scores T {t} Labels {labels[i]}")
    
    scores_r = np.array(scores_r)
    scores_r = normalize(scores_r)

    scores_t = np.array(scores_t)
    scores_t = normalize(scores_t)

    # for i,score_tuple in enumerate(zip(scores_r, scores_t)):
    #     r,t = score_tuple
    #     print(f"Frame {i} Scores R {r} Scores T {t} Labels {labels[i]}")

    scores = scores_r + scores_t
    
    # print(f"Scores {scores} Labels {labels}")
    # for i,(s,l) in enumerate(zip(scores, labels)):
    #     print(f"Frames {i} Scores {s} Labels {l}")
    
    key, value = sample_id, {"scores":scores, "labels":labels}

    #print(f"Labels {len(labels)} Scores {len(result_scores_dict.keys())}")

    #exit()

    return key, value

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()

    config = ConfigParser(args.config_file).config

    checkpoint = config['saved_model']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_path = config['test_ann_path']

    annotation_files = [os.path.join(root, file) for root,dir,files in os.walk(test_path) for file in files if file.endswith(".csv")]
    annotation_files.sort()

    model = load_model(config, device=device)
    
    result_dict = {}
    for ann_file in annotation_files:
        print(f"Evaluating {ann_file}")
        key, value = compute_anomaly_score_per_video_v2(model=model, config=config, ann_file=ann_file)
        if not key in result_dict.keys():
            result_dict[key] = value
        #break

    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    model_name = os.path.basename(checkpoint).replace('.pth','')
    dest_folder = "./results"
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    padding_mode = config.get('padMode','orig')
    dest_file = os.path.join(dest_folder,f"results_raw_{model_name}_{config['seqLen']}_frames_pad_{padding_mode}_v2.npz")
    
    print(f"Saving results from model {model_name} to {dest_file}")
    np.savez(dest_file,**result_dict)

    macro_auc_values = extract_metrics(dest_file,padding=padding_mode)

    mean_auc = 0.0
    for key in macro_auc_values.keys():
        macro_auc = macro_auc_values[key]
        mean_auc += macro_auc

    mean_auc = mean_auc/len(macro_auc_values.keys())
    print(f"Mean AUC {mean_auc}")

        


