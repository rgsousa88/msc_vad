import torch
import cv2

from yolo_inference import YOLOInference, DetectionResult

import os
import gc
from time import time

base_dir = "/mnt/c/dataset/MSC/shanghai_base"
base_dest = "/mnt/c/dataset/MSC/shanghai/"

base_path_train = os.path.join(base_dir, "train_frames")
base_path_val = os.path.join(base_dir, "val_frames")
base_path_test = os.path.join(base_dir, "testing/frames")

base_dest_path_train = os.path.join(base_dest, "train_crop")
base_dest_path_val = os.path.join(base_dest, "val_crop")
base_dest_path_test = os.path.join(base_dest, "test_crop")

    
def detect(model, sample_dir:str, base_dest_dir:str, window:int=2):
    files = [os.path.join(sample_dir, file) for file in os.listdir(sample_dir) if file.endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif"))]
    if len(files) == 0:
        return
    
    files.sort()
    start = window

    if "test" in sample_dir:
        start = 0
        base_sample_dest_dir = os.path.join(base_dest_dir, os.path.basename(sample_dir))
        if not os.path.exists(base_sample_dest_dir):
            print(f"Creating Test Dest Dir {base_sample_dest_dir}")
            os.makedirs(base_sample_dest_dir)
    else:
        base_sample_dest_dir = base_dest_dir

    
    with torch.no_grad():
        tik = time()
        for i in range(start, len(files)-start, 1):
            base_file = os.path.splitext(os.path.basename(files[i]))[0]
            base_sample_id = base_file.rsplit("_",1)[0]
            
            dest_dir = os.path.join(base_sample_dest_dir, base_sample_id)
            
            if not os.path.exists(dest_dir):
                print(f"Creating {dest_dir}")
                os.makedirs(dest_dir)

            print(f"Base File {base_file} Base Sample ID {base_sample_id}")

            res = model(files[i])
            preds = [DetectionResult(pred) for pred in res]
            if len(preds) == 0:
                print(f"No Detection Found on {files[i]}")
                continue
            crops = {}

            for j in range(i-window,i+window+1):
                j_index = min(max(0,j),len(files)-1)
                img = cv2.imread(files[j_index])
                for k, det in enumerate(preds):
                    crop = img[int(det.y_min):int(det.y_max),int(det.x_min):int(det.x_max),:]
                    if not k in crops.keys():
                        crops[k] = []
                    crops[k].append(crop)
                    del crop
                del img
            
            for crop_id in crops.keys():
                crops_by_id = crops[crop_id]
                for crop_idx, crop in enumerate(crops_by_id):
                    dest_filename = os.path.join(dest_dir, f"{base_file}_{crop_id}_{crop_idx}.png")
                    cv2.imwrite(dest_filename, crop)

            del crops
            del preds
            del res
            #break
    del files
    
    gc.collect()
    print(f"Elapsed time {time() - tik:.3f}")

def main(base_path: str, base_dest_path: str, window=2):
    assert os.path.exists(base_path) == True

    sample_dirs_val = [os.path.join(base_path, dir) for dir in os.listdir(base_path)]
    sample_dirs_val.sort()
    print(len(sample_dirs_val))
    print(sample_dirs_val[0:5])

    yolo_model = YOLOInference(model_version="yolov5", model_path="yolov5m.pt", conf_threshold=0.80)

    base_dest_path = f"{base_dest_path}_{window}"

    for i in range(len(sample_dirs_val)):
        print(f"Detecting in folder  {sample_dirs_val[i]}")
        detect(yolo_model, sample_dirs_val[i], base_dest_path, window=window)
        #break
    
    del yolo_model
    gc.collect()

if __name__ == "__main__":
    main(base_path=base_path_val, base_dest_path=base_dest_path_val, window=4)

