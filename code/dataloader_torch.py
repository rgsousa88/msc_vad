import numpy as np
import os
import cv2

import torch
from torch.utils.data import Dataset

torch.manual_seed(122344)

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