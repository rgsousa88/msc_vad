# %%
import numpy as np
import cv2
import os, sys
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

from torchvision.transforms import v2
from torch.utils.data import DataLoader

# %%
class MaskedDataset(Dataset):
    def __init__(self, root_dir, inputSize:tuple[int, int], transform=None, coverFactor:float = 0.3, squareSize = 3, coverMethod:str = "random"):
        self.root_dir = root_dir
        self.__allow_list = (".jpg", ".jpeg", ".png", ".tiff", ".tif")
        self.files = [os.path.join(root, file) for root,dirs,files in os.walk(root_dir) for file in files if file.endswith(self.__allow_list)]
        self.transform = transform
        self.coverFactor = coverFactor
        self.squareSize = squareSize
        self.coverMethod = coverMethod
        self.inputHeight = inputSize[0]
        self.inputWidth = inputSize[1]
        self.nSquares = int(self.coverFactor * (self.inputHeight * self.inputWidth) / (self.squareSize * self.squareSize))
    
    def load_image_as_torch(self, filename: str):
        img = cv2.imread(filename, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)

        return img

    def __len__(self,):
        return len(self.files)
    
    def __getitem__(self, index):
        filePath = self.files[index]
        img = self.load_image_as_torch(filePath)

        if self.transform:
            img = self.transform(img)

        c, h, w = img.shape[0], img.shape[1], img.shape[2]

        if self.inputHeight != h or self.inputWidth != w:
            raise Exception
        
        maskedImg = img.clone()
        coin = torch.randint(low=0, high=2, size=(1,))
        
        if coin[0] == 0:
            rows = torch.randint(low=0, high=h-self.squareSize,size=(self.nSquares,),dtype=torch.int32)
            cols = torch.randint(low=0, high=w-self.squareSize,size=(self.nSquares,),dtype=torch.int32)

            if self.coverMethod == "ones":
                cover = torch.ones((self.nSquares, self.squareSize, self.squareSize))
            elif self.coverMethod == "zeros":
                cover = torch.zeros((self.nSquares, self.squareSize, self.squareSize))
            elif self.coverMethod == "gray":
                cover = 0.5 * torch.ones((self.nSquares, self.squareSize, self.squareSize))
            elif self.coverMethod == "random":
                cover = torch.rand((self.nSquares, c, self.squareSize, self.squareSize))

            for i in range(self.nSquares):
                maskedImg[:,rows[i]:rows[i]+self.squareSize,cols[i]:cols[i]+self.squareSize] = cover[i,:]
            
        return maskedImg, img 

def view_samples(samples):
    nSamples = samples.shape[0]
    for i in range(nSamples):
        sample = samples[i]
        sample = sample.permute(1,2,0).detach().cpu().numpy()
        print(sample.shape)

        plt.imshow(sample, cmap='gray')
        plt.show()


if __name__ == "__main__":
    DATASET_PATH = ""
    inputSize = 128

    transf = v2.Compose([v2.Resize(size=(inputSize, inputSize), antialias=True),
                        v2.ToDtype(torch.float32, scale=True),])
    dataset = MaskedDataset(DATASET_PATH, transform=transf, squareSize=5)
    loader = DataLoader(dataset=dataset,
                        batch_size=5,
                        shuffle=True)
    
    for batch in loader:
        img, masked = batch[0], batch[1]
        print(img.shape)
        print(masked.shape)

        view_samples(img)
        view_samples(masked)
        break


