from nvidia.dali import pipeline_def, pipeline
from nvidia.dali.pipeline import do_not_convert
import nvidia.dali.fn as fn
import nvidia.dali.types as types

image_dir = "/mnt/c/dataset/archive/Fruits/valid"
max_batch_size = 8

@pipeline_def(num_threads=1, device_id=0)
def simple_pipeline(path: str):
    images, labels = fn.readers.file(file_root=path)
    images = fn.decoders.image(images, output_type=types.RGB, name='image')
    images = fn.resize(images.gpu(), resize_x=224, resize_y=224, device='gpu')
    images = fn.normalize(images)

    return images, labels.gpu()

@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0)
def video_seq_pipe(file_root, shape=(224,224), train=True, device='gpu', sequence_length=10, fstride=2, cstride=20, change_color_prob=0.3):
    video = fn.readers.sequence(file_root=file_root,
                                sequence_length=sequence_length,
                                stride=fstride,
                                step=cstride,
                                name='seq')
    
    video = fn.resize(video.gpu(), size=shape, device=device)
    
    if train:
        mirror = fn.random.choice(2)
    else:
        change_color_prob = 0.0
        mirror = 0
    
    video = fn.crop_mirror_normalize(video, dtype=types.FLOAT, std=[255.0], mirror=mirror, output_layout="FHWC")
    do_color_changes = fn.random.coin_flip(probability=change_color_prob, dtype=types.DALIDataType.BOOL)

    if do_color_changes:
        randBrightness = fn.uniform(range=(0.5,1.5))
        randContrast = fn.uniform(range=(0.5,1.5))
        video = fn.brightness_contrast(video, brightness=randBrightness, contrast=randContrast)
    else:
        video = video
    
    video = fn.transpose(video, perm=[3,0,1,2])

    return video

@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0)
def video_pipe(file_root, shape=(224,224), train=True, device='gpu', sequence_length=10, stride=2, step=10, change_color_prob=0.3, initial_pref=512):
    video, labels = fn.readers.video(device=device,
                                    file_root=file_root,
                                    sequence_length=sequence_length,
                                    stride=stride,
                                    step=step,
                                    random_shuffle=True,
                                    initial_fill=initial_pref, name='seq')
    
    video = fn.resize(video.gpu(), size=shape, device=device)
    
    if train:
        mirror = fn.random.choice(2)
    else:
        change_color_prob = 0.0
        mirror = 0
    
    video = fn.crop_mirror_normalize(video, dtype=types.FLOAT, std=[255.0], mirror=mirror, output_layout="FHWC")
    do_color_changes = fn.random.coin_flip(probability=change_color_prob, dtype=types.DALIDataType.BOOL)

    if do_color_changes:
        randBrightness = fn.uniform(range=(0.5,1.5))
        randContrast = fn.uniform(range=(0.5,1.5))
        video = fn.brightness_contrast(video, brightness=randBrightness, contrast=randContrast)
    
    video = fn.transpose(video[1:,:,:,:], perm=[3,0,1,2])

    labels = fn.cast(labels, dtype=types.INT64)

    return video, labels.gpu()

import csv
from random import shuffle
import numpy as np

@do_not_convert
class ExternalInputIterator(object):
    def __init__(self, batch_size, csv_file):
        self.batch_size = batch_size
        self.files = []
        with open(csv_file, 'r') as f:
            for row in csv.reader(f, delimiter=';'):
                if row:
                    self.files.append(row)
        shuffle(self.files)
        self.full_iterations = len(self.files) // self.batch_size

        self.motion_idx = [[0,1,3,4,5,6,8],[0,1,3,4,5,7,8],[0,1,3,4,5,6,7],[0,1,3,4,6,7,8],
                           [0,2,3,4,5,6,8],[0,2,3,4,5,7,8],[0,2,3,4,5,6,7],[0,2,3,4,6,7,8],
                           [1,2,3,4,5,7,8],[1,2,3,4,5,6,8],[0,2,3,4,6,7,8],
                           [0,1,2,4,5,6,8],[0,1,2,4,5,7,8],[0,1,2,4,5,6,7],[0,1,2,4,6,7,8]]

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()
        
        batch = []
        
        row = self.files[sample_idx]
        num_files = len(row)
        for img_path in row[2:num_files]:
            with open(img_path, "rb") as f:
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
        
        resnet = np.squeeze(np.load(row[0]).astype(np.float32))
        yolo = np.load(row[1]).astype(np.float32)
        
        batch.append(resnet)
        batch.append(yolo)
        batch.append(np.array(self.motion_idx).astype(np.int32))

        return batch

@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0, batch_size=4)
def ssmtl_pipe(ann_file, num_frames, batch_size, shape=(64,64), train=True, device='gpu', arrow_prob=0.5, motion_prob=0.5):
    *jpegs, resnet, yolo, motion_idx = fn.external_source(source=ExternalInputIterator(csv_file=ann_file, batch_size=batch_size),
                               num_outputs=num_frames+3,
                               batch=False)
    images = fn.decoders.image(jpegs, device="mixed")
    
    sequence = fn.resize(images, size=shape, device=device)
    sequence = fn.stack(*sequence)
    sequence = fn.reshape(sequence, layout="FHWC")

    if train:
        arrow_prob = 0.5
        motion_prob = 0.5
        recon_idx = types.Constant(np.array([1,2,3,5,6,7]), shape=(6,), dtype=types.DALIDataType.INT32)
        middle_frame = 4
    else:
        arrow_prob = 0.0
        motion_prob = 0.0
        recon_idx = types.Constant(np.array([0,1,2,4,5,6]), shape=(6,), dtype=types.DALIDataType.INT32)
        middle_frame = 3
    
    do_backward = fn.random.coin_flip(probability=arrow_prob, dtype=types.DALIDataType.BOOL)
    do_motion = fn.random.coin_flip(probability=motion_prob, dtype=types.DALIDataType.BOOL)

    if train:
        seq_backward = sequence[1:-1,:]
        seq_motion = sequence[1:-1,:]
    else:
        seq_backward = sequence
        seq_motion = sequence

    label_backward = fn.zeros(shape=1)

    if do_backward:
        seq_backward = fn.flip(seq_backward, depthwise=1, horizontal=0, vertical=0)
        label_backward = fn.ones(shape=1)
    
    label_sequence = fn.zeros(shape=1)

    if do_motion:
        idx = fn.random.choice(15)
        seq_motion = fn.sequence_rearrange(sequence, new_order=motion_idx[idx])
        label_sequence = fn.ones(shape=1)
    
    seq_recon = fn.sequence_rearrange(sequence, new_order=recon_idx)
    seq_distill = sequence[middle_frame,:]

    features = fn.cat(resnet, yolo, axis=0)

    return seq_backward, label_backward.gpu(), seq_motion, label_sequence.gpu(), seq_recon, seq_distill, features.gpu()



if __name__ == "__main__":
    pipe = simple_pipeline(image_dir, batch_size=max_batch_size, num_threads=1, device_id=0)
    pipe.build()



    

