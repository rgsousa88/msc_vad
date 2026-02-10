import torch
import cv2
import numpy as np

from yolo_inference import YOLOInference, DetectionResult
from resnet_encoder import ResNet50Encoder

import os
import gc
from time import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial


base_dir = "/mnt/c/dataset/MSC/shanghai_base"
base_dest = "/mnt/c/dataset/MSC/shanghai/"

base_path_train = os.path.join(base_dir, "train_frames_2")
base_path_val = os.path.join(base_dir, "val_frames_2")
base_path_test = os.path.join(base_dir, "testing/frames")

base_dest_path_train = os.path.join(base_dest, "train_crop")
base_dest_path_val = os.path.join(base_dest, "val_crop")
base_dest_path_test = os.path.join(base_dest, "test_crop")


def detect(yolo_model, resnet_model, ann_file, sample_dir:str, base_dest_dir:str, window:int=2):
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
            dest_dir_logits = os.path.join(base_sample_dest_dir, base_sample_id, "resnet_logits")
            dest_dir_yolo_probs = os.path.join(base_sample_dest_dir, base_sample_id, "yolo_probs")
            
            if not os.path.exists(dest_dir):
                print(f"Creating {dest_dir}")
                os.makedirs(dest_dir)
            # else:
            #     return
            
            if not os.path.exists(dest_dir_logits):
                print(f"Creating {dest_dir_logits}")
                os.makedirs(dest_dir_logits)

            if not os.path.exists(dest_dir_yolo_probs):
                print(f"Creating {dest_dir_yolo_probs}")
                os.makedirs(dest_dir_yolo_probs)

            #print(f"Base File {base_file} Base Sample ID {base_sample_id}")

            res = yolo_model.do_inference(files[i])
            preds = [DetectionResult(pred) for pred in res]
            if len(preds) == 0:
                #print(f"No Detection Found on {files[i]}")
                continue
            crops = {}

           
            for j in range(i-window,i+window+1):
                j_index = min(max(0,j),len(files)-1)
                
                img = cv2.imread(files[j_index])
                for k, det in enumerate(preds):
                    crop = img[int(det.y_min):int(det.y_max),int(det.x_min):int(det.x_max),:]
                    if not k in crops.keys():
                        crops[k] = {'crops':[]}
                    
                    if j_index == i:
                        logits = resnet_model.do_inference(crop)
                        crops[k]['logits'] = logits
                        crops[k]['probs'] = np.array(det.probs)
                        del logits
                    
                    crops[k]['crops'].append(crop)
                    del crop
                del img
            
            
            line = ""
            for crop_id in crops.keys():
                crops_by_id = crops[crop_id]['crops']
                dest_files_list = []
                for crop_idx, crop in enumerate(crops_by_id):
                    dest_filename = os.path.join(dest_dir, f"{base_file}_{crop_id}_{crop_idx}.png")
                    dest_files_list.append(dest_filename)
                    cv2.imwrite(dest_filename, crop)
                
                if (not crops[crop_id]['logits'] is None) and (not crops[crop_id]['probs'] is None):
                    dest_filename_logits = os.path.join(dest_dir_logits, f"{base_file}_{crop_id}.npy")
                    dest_filename_probs = os.path.join(dest_dir_yolo_probs, f"{base_file}_{crop_id}.npy")

                    np.save(dest_filename_logits, crops[crop_id]['logits'])
                    np.save(dest_filename_probs, crops[crop_id]['probs'])

                    line = f"{dest_filename_logits};{dest_filename_probs};{";".join(dest_files_list)}\n"
                    ann_file.write(line)

            del crops
            del preds
            del res
            #break
    del files
    
    gc.collect()
    print(f"Elapsed time {time() - tik:.3f}")

def process_single_directory(yolo_model, resnet50_model, base_dest_path, window, sample_dir):
    """Processa um único diretório - função para ser executada em thread"""
    dest_ann_file = os.path.join(base_dest_path, f"annotation_{os.path.basename(sample_dir)}.csv")
    
    with open(dest_ann_file, 'w') as ann_file:
        print(f"Detecting in folder {sample_dir}")
        detect(yolo_model, resnet50_model, ann_file, sample_dir, base_dest_path, window=window)
    
    return sample_dir

def extract_with_threads(base_path: str, base_dest_path: str, window=2, num_threads=4):
    assert os.path.exists(base_path) == True

    sample_dirs_val = [os.path.join(base_path, dir) for dir in os.listdir(base_path)]
    sample_dirs_val.sort()
    print(f"Total directories to process: {len(sample_dirs_val)}")
    print(f"First 5 directories: {sample_dirs_val[0:5]}")

    # Inicializa os modelos uma vez (compartilhados entre threads)
    yolo_model = YOLOInference(model_version="yolov5", model_path="yolov5m.pt", conf_threshold=0.80)
    resnet50_model = ResNet50Encoder(device=yolo_model.device)

    base_dest_path = f"{base_dest_path}_{window}"
    if not os.path.exists(base_dest_path):
        os.makedirs(base_dest_path)

    # Cria uma versão parcial da função com argumentos fixos
    process_func = partial(process_single_directory, 
                          yolo_model, 
                          resnet50_model, 
                          base_dest_path, 
                          window)

    # Processa em paralelo com pool de threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submete todas as tarefas
        future_to_dir = {
            executor.submit(process_func, dir): dir 
            for dir in sample_dirs_val
        }
        
        # Processa resultados conforme completam
        for future in as_completed(future_to_dir):
            sample_dir = future_to_dir[future]
            try:
                result = future.result()
                print(f"Completed processing: {result}")
            except Exception as e:
                print(f"Error processing {sample_dir}: {e}")

    # Limpeza
    del yolo_model
    del resnet50_model
    gc.collect()

def main(base_path: str, base_dest_path: str, window=2):
    assert os.path.exists(base_path) == True

    sample_dirs_val = [os.path.join(base_path, dir) for dir in os.listdir(base_path)]
    sample_dirs_val.sort()
    print(len(sample_dirs_val))
    print(sample_dirs_val[0:5])

    yolo_model = YOLOInference(model_version="yolov5", model_path="yolov5m.pt", conf_threshold=0.80)
    resnet50_model = ResNet50Encoder(device=yolo_model.device)

    base_dest_path = f"{base_dest_path}_{window}"
    if not os.path.exists(base_dest_path):
        os.makedirs(base_dest_path)

    for i in range(len(sample_dirs_val)):
        dest_ann_file = os.path.join(base_dest_path, f"annotation_{os.path.basename(sample_dirs_val[i])}.csv")
        with open(dest_ann_file, 'w') as ann_file:
            print(f"Detecting in folder  {sample_dirs_val[i]}")
            detect(yolo_model, resnet50_model, ann_file, sample_dirs_val[i], base_dest_path, window=window)
            break
    
    del yolo_model
    del resnet50_model
    gc.collect()

if __name__ == "__main__":
    #main(base_path=base_path_val, base_dest_path=base_dest_path_val, window=4)
    extract_with_threads(base_path=base_path_train, base_dest_path=base_dest_path_train, window=4, num_threads=8)

