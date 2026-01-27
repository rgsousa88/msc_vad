import numpy as np
import os
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import pandas as pd
import random

SEED = 122344
random.seed(SEED)
torch.manual_seed(SEED)

class ShanghaiTestSampleDataset(Dataset):
    def __init__(self, video_path, label_path, seqLen=10, cstride=10, fstride=1, transform=None, encoding=True):
        self.root_dir = video_path
        self.labels_dir = label_path
        self.seqlen = seqLen
        self.fstride = fstride
        self.cstride = cstride
        self.transform = transform
        self.__allow_list = (".jpg", ".jpeg", ".png", ".tiff", ".tif")
        self.encoding = encoding

        assert self.cstride == self.seqlen * self.fstride

        self.clips = []
        self.labels = []

        frame_files = [os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if file.endswith(self.__allow_list)]
        frame_files.sort()
        num_frames = len(frame_files)
        labels = np.load(self.labels_dir)

        for start in range(0, num_frames - (self.seqlen * self.fstride) + 1, self.cstride):
            end = start + self.seqlen * self.fstride
            clip = frame_files[start:end:self.fstride]
            label = labels[start:end:self.fstride]
            self.clips.append(clip)
            self.labels.append(label)
    
    def load_image_as_torch(self, filename: str):
        frame = cv2.imread(filename, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame)
        frame = frame.permute(2,0,1)

        return frame
    
    def __len__(self,):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clips_files = self.clips[idx]
        labels = self.labels[idx]

        frames = [self.load_image_as_torch(frameFile) for frameFile in clips_files]
        frames = torch.stack(frames, dim=0)
        
        if self.transform:
            frames = self.transform(frames)
        
        if self.encoding:
            frames = frames.permute(1,0,2,3)
        
        return frames, labels.astype('int64')


class ShanghaiTestDataset(Dataset):
    def __init__(self, videos_dir, labels_dir, seqLen=10, cstride=10, fstride=1, transform=None, overlapping=False, encoding=True):
        self.root_dir = videos_dir
        self.labels_dir = labels_dir
        self.seqlen = seqLen
        self.fstride = fstride
        self.transform = transform
        self.__allow_list = (".jpg", ".jpeg", ".png", ".tiff", ".tif")
        self.encoding = encoding

        self.cstride = cstride if overlapping else self.seqlen * self.fstride

        self.video_folders = [os.path.join(self.root_dir, folder) for folder in os.listdir(self.root_dir)]    
        self.video_folders.sort()

        self.label_files = [os.path.join(self.labels_dir, file) for file in os.listdir(self.labels_dir) if file.endswith('.npy')]
        self.label_files.sort()

        self.clips = []
        self.labels = []

        for i,video_folder in enumerate(self.video_folders):
            frame_files = [os.path.join(video_folder, file) for file in os.listdir(video_folder) if file.endswith(self.__allow_list)]
            frame_files.sort()
            num_frames = len(frame_files)
            labels = np.load(self.label_files[i])

            for start in range(0, num_frames - (self.seqlen * self.fstride) + 1, self.cstride):
                end = start + self.seqlen * self.fstride
                clip = frame_files[start:end:self.fstride]
                label = labels[start:end:self.fstride]
                self.clips.append(clip)
                self.labels.append(label)      
    
    def load_image_as_torch(self, filename: str):
        frame = cv2.imread(filename, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame)
        frame = frame.permute(2,0,1)

        return frame

    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clips_files = self.clips[idx]
        labels = self.labels[idx]

        frames = [self.load_image_as_torch(frameFile) for frameFile in clips_files]
        frames = torch.stack(frames, dim=0)
        
        if self.transform:
            frames = self.transform(frames)
        
        if self.encoding:
            frames = frames.permute(1,0,2,3)
        
        return frames, labels.astype('int64')
    
class SSMTLModelDataset(Dataset):
    def __init__(self, annotation_path, input_size=64, window=4, transform=None, is_test=False):
        self.annotation_path = annotation_path
        self.test_mode = is_test
        self.window = window
        self.num_frames_per_obj = 2 * window + 1
        self.middle_frame_index = self.num_frames_per_obj // 2

        if not is_test:
            self.recon_indices = list(range(1,self.middle_frame_index)) + list(range(self.middle_frame_index+1,self.num_frames_per_obj-1))
        else:
            self.recon_indices = list(range(self.middle_frame_index)) + list(range(self.middle_frame_index+1,self.num_frames_per_obj))
        self.recon_indices = sorted(self.recon_indices)
        self.indices_before = list(range(0, self.middle_frame_index))
        self.indices_after = list(range(self.middle_frame_index + 1, self.num_frames_per_obj))

        self.input_shape = (input_size, input_size)
        self.col_names = ['resnet_logits','yolo_prob'] + [f"object_{i}" for i in range(self.num_frames_per_obj)]
        self.df_ann = pd.read_csv(annotation_path, sep=';', names=self.col_names)

        self.transform =[]
        if not transform is None:
            self.transform.extend(transform)
        
        self.transform.extend([v2.Resize(size=self.input_shape, antialias=True),
                               v2.ToDtype(torch.float32, scale=True),])
        
        self.transform = v2.Compose(self.transform)

    def __len__(self,):
        return len(self.df_ann)
    
    def load_and_preprocessing(self, img_path:str):
        frame = cv2.imread(img_path, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame)
        frame = frame.permute(2,0,1)
        frame = self.transform(frame)

        return frame
    
    def slice_motion_tensor(self, x):
        selected_before = random.sample(self.indices_before, 3)
        selected_after = random.sample(self.indices_after, 3)

        selected_indices = sorted(selected_before + [self.middle_frame_index] + selected_after)
        
        return x[:, selected_indices, :, :]
    
    def create_inputs(self, x):
        C,N,H,W = x.shape

        x_arrow = x[:,:,:,:]
        l_arrow = torch.tensor(0, dtype=torch.int64)

        x_motion = x[:,:,:,:]
        l_motion = torch.tensor(0, dtype=torch.int64)

        if not self.test_mode:
            x_arrow = x[:,1:-1,:,:]
            x_motion = x[:,1:-1,:,:]

            if random.randint(0, 1) == 1: #0 -> forward, 1 -> backward
                x_arrow = torch.flip(x_arrow, dims=[1])
                l_arrow = torch.tensor(1, dtype=torch.int64)

            if random.randint(0, 1) == 1:
                x_motion = self.slice_motion_tensor(x)
                l_motion = torch.tensor(1, dtype=torch.int64)
        
        x_recon = x[:,self.recon_indices,:,:]
        x_distil = x[:,self.middle_frame_index,:,:].reshape(C,-1,H,W)

        return x_arrow, l_arrow, x_motion, l_motion, x_recon, x_distil
    
    def __getitem__(self, idx):
        items = self.df_ann.iloc[idx]
        resnet_logits = np.load(items[self.col_names[0]])
        yolo_probs = np.load(items[self.col_names[1]])

        img_crops = [self.load_and_preprocessing(items[self.col_names[i]]) for i in range(2, len(self.col_names))]
        img_crops = torch.stack(img_crops, dim=0)
        img_crops = img_crops.permute(1,0,2,3) #C,F,H,W

        feat_distil = torch.cat((torch.from_numpy(resnet_logits).squeeze(), torch.from_numpy(yolo_probs)), dim=0)
        x_arrow, l_arrow, x_motion, l_motion, x_recon, x_distil = self.create_inputs(img_crops)
        
        if self.test_mode:
            logit_path = items[self.col_names[0]]
            dirname = os.path.dirname(os.path.dirname(logit_path))
            frame_id = os.path.basename(dirname)
            dirname = os.path.dirname(dirname)
            sample_id = os.path.basename(dirname)
            key = f"{sample_id}_{frame_id}"

            return (x_arrow, l_arrow), (x_motion, l_motion), x_recon, (x_distil, feat_distil), key
        
        return (x_arrow, l_arrow), (x_motion, l_motion), x_recon, (x_distil, feat_distil)


        