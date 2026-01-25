from torchvision.models import resnet50, ResNet50_Weights

import torch
import cv2
import numpy as np

import gc

class ResNet50Encoder:
    def __init__(self, input_size=64, device='cpu'):
        self.input_size = input_size
        self.device = device
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.eval()
        self.model = self.model.to(device)
    
    def preprocessing(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #BGR -> RGB
        img = cv2.resize(img, (self.input_size, self.input_size))

        img = img.astype('float32') / 255.0
        img = np.transpose(img, (2,0,1)) #HWC -> CHW
        img = np.expand_dims(img, axis=0) #CHW -> BCHW

        img_tensor = torch.from_numpy(img)
        return img_tensor
    
    def do_inference(self, img):
        with torch.no_grad():
            input_tensor = self.preprocessing(img)
            logits = self.model(input_tensor.to(self.device))
        
        del input_tensor
        torch.cuda.empty_cache()
        gc.collect()

        return logits.detach().cpu().numpy()