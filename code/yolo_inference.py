import torch
import numpy as np
import cv2

import gc

class DetectionResult():
    def __init__(self, pred): # pred is a tensor
        if isinstance(pred, torch.Tensor):
            preds = pred.detach().cpu().numpy()

            self.x_min = preds[0]
            self.y_min = preds[1]
            self.x_max = preds[2]
            self.y_max = preds[3]
            self.score = preds[4]
            self.class_id = preds[5]
            self.probs = []

        elif isinstance(pred, dict):
            self.x_min = pred['bbox'][0]
            self.y_min = pred['bbox'][1]
            self.x_max = pred['bbox'][2]
            self.y_max = pred['bbox'][3]
            self.score = pred['confidence']
            self.class_id = pred['class_id']
            self.probs = pred['probabilities']
        
        else:
            raise ValueError(f"Invalid pred type {type(pred)}")
    
    def __str__(self,):
        msg = f"(x_min,y_min):({self.x_min:.2f},{self.y_min:.2f}) (x_max,y_max):({self.x_max:.2f},{self.y_max:.2f})"
        msg = f"{msg} score: {self.score:.4f} class_id {self.class_id}"

        return msg

class YOLOInference:
    def __init__(self, model_version:str, model_path:str, conf_threshold:float = 0.5, iou_threshold:float = 0.5, device:str='cuda'):
        self.model_version = model_version
        self.model_path = model_path

        self.yolo = torch.hub.load(f"ultralytics/{self.model_version}", model='custom', path=self.model_path, verbose=False, autoshape=False)
        self.device = device
        self.yolo = self.yolo.to(device)
        self.img_size = (640, 640)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """
        Redimensiona com padding mantendo aspect ratio
        """
        # Dimensões originais
        shape = im.shape[:2]
        
        # Redimensionar para nova forma
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Escala (ratio)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        
        # Calcular padding
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        
        # Dividir padding igualmente
        dw /= 2
        dh /= 2
        
        # Redimensionar e adicionar padding
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, 
                            cv2.BORDER_CONSTANT, value=color)
        
        return im, ratio, (dw, dh)
    
    def preprocessing(self, img):
        # img -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #BGR -> RGB
        img = self.letterbox(img, new_shape=self.img_size, auto=True)[0]
        print(f"Input image resized to {img.shape}")

        img = img.astype('float32') / 255.0
        img = np.transpose(img, (2,0,1)) #HWC -> CHW
        img = np.expand_dims(img, axis=0) #CHW -> BCHW

        img_tensor = torch.from_numpy(img)
        return img_tensor
    
    def nms_coco_output(self, pred):
        """
        Aplica Non-Maximum Suppression (NMS) ao output do YOLOv5 e retorna
        bounding boxes com probabilidades das 80 classes do COCO.
        
        Args:
            output: Output do modelo YOLOv5
        
        Returns:
            Lista de dicionários com bounding boxes e probabilidades
        """
        results = []
        
        # [batch_size, num_detections, 85]
        # 85 = [xc, yc, w, h, confidence, class_probs...]
        
        if len(pred.shape) == 3:
            pred = pred[0]
        
        if pred.shape[1] > 5:
            confidence_scores = pred[:, 4]
            mask = confidence_scores > self.conf_threshold
            pred = pred[mask]
        
        if len(pred) == 0:
            return results
        
        boxes = pred[:, :4]  # xc, yc, w, h
        scores = pred[:, 4]  # confidence scores

        new_boxes = torch.zeros_like(boxes) #x1,y1,x2,y2
        new_boxes[:,0] = boxes[:,0]-(boxes[:,2]/2)
        new_boxes[:,1] = boxes[:,1]-(boxes[:,3]/2)
        new_boxes[:,2] = boxes[:,0]+(boxes[:,2]/2)
        new_boxes[:,3] = boxes[:,1]+(boxes[:,3]/2)

        class_probs = pred[:, 5:]  # Probabilidades das 80 classes do COCO
        class_ids = torch.argmax(class_probs, dim=1)
        class_scores = scores * class_probs[torch.arange(len(class_ids)), class_ids]

        # Aplicar NMS
        keep_indices = torch.ops.torchvision.nms(new_boxes, class_scores, self.iou_threshold)
        
        # Formatar resultados
        for idx in keep_indices:
            box = new_boxes[idx].cpu().numpy()
            class_id = class_ids[idx].item()
            confidence = class_scores[idx].item()
            probabilities = class_probs[idx].cpu().numpy()
            
            result = {
                'bbox': box.tolist(), #[x1, y1, x2, y2]
                'class_id': class_id,
                'confidence': confidence,
                'probabilities': probabilities.tolist(),
            }
            results.append(result)
        
        return results
    
    def convert_to_original_coords(self, bbox, original_shape, resized_shape):
        """
        Converte detecções para coordenadas da imagem original
        
        Args:
            detections: Lista de [x1, y1, x2, y2] na imagem redimensionada
            original_shape: (altura, largura) da imagem original
            resized_shape: Tamanho da imagem após letterbox
        """
        orig_h, orig_w = original_shape[0], original_shape[1]
        
        # Calcular ratio e padding
        gain = min(resized_shape[-2]/orig_h, resized_shape[-1]/orig_w)
        pad_x = (resized_shape[-1] - orig_w * gain) / 2
        pad_y = (resized_shape[-2] - orig_h * gain) / 2
        
        x1, y1, x2, y2, = bbox
        
        # Remover padding
        x1 = (x1 - pad_x) / gain
        y1 = (y1 - pad_y) / gain
        x2 = (x2 - pad_x) / gain
        y2 = (y2 - pad_y) / gain
        
        # Clip para limites
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        
        fixed_bbox = [x1, y1, x2, y2]
        
        return fixed_bbox


    def do_inference(self, img):
        if isinstance(img, str):
            input_img = cv2.imread(img)
        elif isinstance(self, np.ndarray):
            input_img = img
        else:
            raise AttributeError(f"Invalid input img type {type(img)}")
        
        with torch.no_grad():
            self.yolo.eval()
            original_shape = input_img.shape
            print(f"Original shape {original_shape}")
            img_tensor = self.preprocessing(input_img)
            model_shape = img_tensor.shape
            print(f"Model shape {model_shape}")
            output = self.yolo(img_tensor.to(self.yolo.device))[0]
            result = self.nms_coco_output(output)
        
        for res in result:
            res['bbox'] = self.convert_to_original_coords(res['bbox'], original_shape, model_shape)
        
        del img_tensor
        del output
        torch.cuda.empty_cache()
        gc.collect()

        return result
    
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')

    import matplotlib.pyplot as plt
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform YOLOv5 Object Detection")
    parser.add_argument("--filename", type=str, help="Path to image to be processed")
    parser.add_argument("--use_custom", action="store_true", help="Select custom model to be executed")
    args = parser.parse_args()

    filename = args.filename
    print(f"Filename {filename}")

    use_custom = args.use_custom
    conf_threshold = 0.80
    
    if use_custom:
        yolov5 = YOLOInference(model_version="yolov5", model_path="yolov5m.pt", conf_threshold=conf_threshold)
    else:
        yolov5 = torch.hub.load("ultralytics/yolov5", model='yolov5m', verbose=True, autoshape=True)
        yolov5.conf = conf_threshold

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_copy = img.copy()

    with torch.no_grad():
        if use_custom:
            result = yolov5.do_inference(filename)
        else:
            result = yolov5(filename).pred[0]
        
        for res in result:
            det_res = DetectionResult(res)
            print(f"{det_res}")
            x1, y1, x2, y2 = map(int, (det_res.x_min, det_res.y_min, det_res.x_max, det_res.y_max))
            P1, P2 = (int(x1),int(y1)), (int(x2),int(y2))
            print(f"x1,y1 = ({P1}) x2,y2 = ({P2})")
            img_copy = cv2.rectangle(img_copy, P1, P2, (255,0,0), thickness=4)

    plt.imshow(img_copy)
    plt.show()