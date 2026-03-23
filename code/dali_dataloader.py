from nvidia.dali import pipeline_def, pipeline
from nvidia.dali.pipeline import do_not_convert
import nvidia.dali.fn as fn
import nvidia.dali.types as types

import csv
import random
import numpy as np
import os
import struct

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

@do_not_convert
class ExternalInputIterator(object):
    def __init__(self, batch_size, csv_file, seq_len=7):
        self.batch_size = batch_size
        self.files = []
        with open(csv_file, 'r') as f:
            for row in csv.reader(f, delimiter=';'):
                if row:
                    self.files.append(row)
        random.shuffle(self.files)
        self.full_iterations = len(self.files) // self.batch_size
        self.seq_len = seq_len
        self.p = 0.5
        self.indices = [i for i in range(self.seq_len)]

        # self.motion_idx = [[0,1,3,4,5,6,8],[0,1,3,4,5,7,8],[0,1,3,4,5,6,7],[0,1,3,4,6,7,8],
        #                    [0,2,3,4,5,6,8],[0,2,3,4,5,7,8],[0,2,3,4,5,6,7],[0,2,3,4,6,7,8],
        #                    [1,2,3,4,5,7,8],[1,2,3,4,5,6,8],[0,2,3,4,6,7,8],
        #                    [0,1,2,4,5,6,8],[0,1,2,4,5,7,8],[0,1,2,4,5,6,7],[0,1,2,4,6,7,8]]

        # self.motion_idx = [[0,1,3,4,5,6,8],[0,1,3,4,5,7,8],[0,1,3,4,6,7,8],
        #                    [0,2,3,4,5,6,8],[0,2,3,4,5,7,8],[0,2,3,4,6,7,8],
        #                    [0,1,2,4,5,6,8],[0,1,2,4,5,7,8],[0,1,2,4,6,7,8]]
        
    def irregular_shuffle(self, indices):
        new_indices = indices
        i = random.randint(0, self.seq_len-2)
        new_indices[i], new_indices[i+1] = new_indices[i+1], new_indices[i]

        return new_indices
    
    def irregular_duplicate(self, indices):
        new_indices = indices
        i = random.randint(1, self.seq_len-2)
        new_indices[i] = new_indices[i-1]

        return new_indices
    
    def irregular_timewarp(self,):
        indices = sorted(random.sample(range(self.seq_len), self.seq_len))
        # cria leve distorção temporal
        indices[random.randint(1,self.seq_len-2)] += random.choice([-1,1])
        indices = [max(0, min(self.seq_len-1, i)) for i in indices]

        return indices
    
    def create_motion_sample(self,):
        method = random.choice(["shuffle","duplicate","warp"])
        if method == "shuffle":
            indices = self.irregular_shuffle(indices=self.indices)

        elif method == "duplicate":
            indices = self.irregular_duplicate(indices=self.indices)

        else:
            indices = self.irregular_timewarp()

        return indices
    
    def get_key_prefix_encode(self, resnet_row):
        dirname = os.path.dirname(os.path.dirname(resnet_row))
        frame_id = os.path.basename(dirname)
        dirname = os.path.dirname(dirname)
        sample_id = os.path.basename(dirname)
        key = f"{sample_id}_{frame_id}".encode('utf-8')
        length_prefix = struct.pack('<i', len(key))
        final_payload = length_prefix + key
        enc = np.frombuffer(final_payload, dtype=np.int8)
        enc_prefix = np.frombuffer(length_prefix, dtype=np.int8)

        return enc, enc_prefix


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
        batch.append(np.array(self.create_motion_sample()).astype(np.int32))

        key, prefix = self.get_key_prefix_encode(row[0])
        batch.append(key)
        batch.append(prefix)

        return batch

@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0, batch_size=4)
def ssmtl_pipe(ann_file, num_frames, batch_size, shape=(64,64), train=True, device='gpu', arrow_prob=0.5, motion_prob=0.5):
    *jpegs, resnet, yolo, motion_idx, key, prefix = fn.external_source(source=ExternalInputIterator(csv_file=ann_file, batch_size=batch_size),
                               num_outputs=num_frames+5,
                               batch=False)
    
    images = fn.decoders.image(jpegs, device="mixed")
    
    sequence = fn.resize(images, size=shape, device=device)
    sequence = fn.stack(*sequence)
    sequence = fn.reshape(sequence, layout="FHWC")
    sequence = fn.crop_mirror_normalize(sequence, dtype=types.FLOAT, std=[255.0], output_layout="FHWC")

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

    label_backward = fn.zeros(shape=1, dtype=types.DALIDataType.INT64)

    if do_backward:
        seq_backward = seq_backward[::-1,:,:,:]
        label_backward = fn.ones(shape=1, dtype=types.DALIDataType.INT64)
    
    label_sequence = fn.zeros(shape=1, dtype=types.DALIDataType.INT64)

    if do_motion:
        #idx = fn.random.choice(9)
        seq_motion = fn.sequence_rearrange(sequence, new_order=motion_idx)
        label_sequence = fn.ones(shape=1, dtype=types.DALIDataType.INT64)
    
    seq_recon = fn.sequence_rearrange(sequence, new_order=recon_idx)
    seq_distill = sequence[middle_frame,:] #(H,W,C)
    seq_distill = fn.expand_dims(seq_distill, axes=[0], new_axis_names="F")

    seq_backward = fn.transpose(seq_backward, perm=[3,0,1,2]) #FHWC (0,1,2,3) -> CFHW (3,0,1,2)
    seq_motion = fn.transpose(seq_motion, perm=[3,0,1,2])
    seq_recon = fn.transpose(seq_recon, perm=[3,0,1,2])
    seq_distill = fn.transpose(seq_distill, perm=[3,0,1,2])

    features = fn.cat(resnet, yolo, axis=0)

    return seq_backward, label_backward.gpu(), seq_motion, label_sequence.gpu(), seq_recon, seq_distill, features.gpu(), key.gpu(), prefix.gpu()

from hibrid_mask_dataloader import MaskGeneratorTorch

@do_not_convert
class MaskedInputIterator(object):
    def __init__(self, batch_size, csv_file, seq_len:int=7, input_size:int=64, train:bool=True, cover_factor:float=0.3, square_size:int=3):
        self.batch_size = batch_size
        self.files = []
        with open(csv_file, 'r') as f:
            for row in csv.reader(f, delimiter=';'):
                if row:
                    self.files.append(row)
        random.shuffle(self.files)
        self.full_iterations = len(self.files) // self.batch_size
        self.seq_len = seq_len
        self.is_train = train
        self.p = 0.5
        self.indices = [i for i in range(self.seq_len)]
        self.input_size = input_size
        self.cover_factor = cover_factor
        self.square_size = square_size

        self.mask_generator = MaskGeneratorTorch(height=input_size, width=input_size,
                                                 percent=cover_factor, square_size=square_size,
                                                 device='cpu')
        
        self.n_squares = self.mask_generator.num_squares
        self.radius = self.square_size // 2
        
    def irregular_shuffle(self, indices):
        new_indices = indices
        i = random.randint(0, self.seq_len-2)
        new_indices[i], new_indices[i+1] = new_indices[i+1], new_indices[i]

        return new_indices
    
    def irregular_duplicate(self, indices):
        new_indices = indices
        i = random.randint(1, self.seq_len-2)
        new_indices[i] = new_indices[i-1]

        return new_indices
    
    def irregular_timewarp(self,):
        indices = sorted(random.sample(range(self.seq_len), self.seq_len))
        indices[random.randint(1,self.seq_len-2)] += random.choice([-1,1])
        indices = [max(0, min(self.seq_len-1, i)) for i in indices]

        return indices
    
    def create_motion_sample(self,):
        method = random.choice(["shuffle","duplicate","warp"])
        if method == "shuffle":
            indices = self.irregular_shuffle(indices=self.indices)

        elif method == "duplicate":
            indices = self.irregular_duplicate(indices=self.indices)

        else:
            indices = self.irregular_timewarp()

        return indices
    
    def get_key_prefix_encode(self, resnet_row):
        dirname = os.path.dirname(os.path.dirname(resnet_row))
        frame_id = os.path.basename(dirname)
        dirname = os.path.dirname(dirname)
        sample_id = os.path.basename(dirname)
        key = f"{sample_id}_{frame_id}".encode('utf-8')
        length_prefix = struct.pack('<i', len(key))
        final_payload = length_prefix + key
        enc = np.frombuffer(final_payload, dtype=np.int8)
        enc_prefix = np.frombuffer(length_prefix, dtype=np.int8)

        return enc, enc_prefix


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
        
        num_mask = len(row[2:num_files])

        for _ in range(num_mask):
            mask = np.ones((self.input_size, self.input_size, 3), dtype=np.uint8)
            if self.is_train:
                batch_mask_x, batch_mask_y = self.mask_generator.get_squares_coords()
                for i in range(self.n_squares):
                    r, c = batch_mask_x[i], batch_mask_y[i]
                    mask[r-self.radius:r+self.radius, c-self.radius:c+self.radius,:] = 0
            batch.append(mask)

        key, prefix = self.get_key_prefix_encode(row[0])
        batch.append(key)
        batch.append(prefix)

        return batch
    
@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0, batch_size=4)
def masked_pipe(ann_file, num_frames, batch_size, shape=(64,64), train=True, device='gpu', cover_factor:float=0.3, square_size:int=5):
    *frames, key, prefix = fn.external_source(source=MaskedInputIterator(csv_file=ann_file,
                                                                         batch_size=batch_size,
                                                                         train=train,
                                                                         cover_factor=cover_factor,
                                                                         square_size=square_size),
                               num_outputs=2*num_frames+2,
                               batch=False)
    
    jpegs, masks = frames[:len(frames)//2], frames[len(frames)//2:]
    
    images = fn.decoders.image(jpegs, device="mixed")
    
    sequence = fn.resize(images, size=shape, device=device)
    sequence = fn.stack(*sequence)
    sequence = fn.reshape(sequence, layout="FHWC")
    sequence = fn.crop_mirror_normalize(sequence, dtype=types.FLOAT, std=[255.0], output_layout="FHWC")

    masks = fn.stack(*masks)

    if train:
        prob = 0.9
    else:
        prob = 0.0
    
    do_mask = fn.random.coin_flip(probability=prob, dtype=types.DALIDataType.BOOL)

    if train:
        sequence = sequence[1:-1,:]
        masks = masks[1:-1,:]
    else:
        sequence = sequence
        masks = masks

    if do_mask:
        masked_sequence = (sequence - 0.5) * masks + 0.5
    else:
        masked_sequence = sequence

    sequence = fn.transpose(sequence, perm=[3,0,1,2])
    masked_sequence = fn.transpose(masked_sequence, perm=[3,0,1,2])

    return sequence, masked_sequence.gpu(), key.gpu(), prefix.gpu()

@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0, batch_size=4)
def masked_arrow_pipe(ann_file, num_frames, batch_size, shape=(64,64), train=True, device='gpu', cover_factor:float=0.3, square_size:int=5):
    *frames, key, prefix = fn.external_source(source=MaskedInputIterator(csv_file=ann_file,
                                                                         batch_size=batch_size,
                                                                         train=train,
                                                                         cover_factor=cover_factor,
                                                                         square_size=square_size),
                               num_outputs=2*num_frames+2,
                               batch=False)
    
    jpegs, masks = frames[:len(frames)//2], frames[len(frames)//2:]
    
    images = fn.decoders.image(jpegs, device="mixed")
    
    sequence = fn.resize(images, size=shape, device=device)
    sequence = fn.stack(*sequence)
    sequence = fn.reshape(sequence, layout="FHWC")
    sequence = fn.crop_mirror_normalize(sequence, dtype=types.FLOAT, std=[255.0], output_layout="FHWC")

    masks = fn.stack(*masks)

    if train:
        prob = 0.9
        prob_backward = 0.5
    else:
        prob = 0.0
        prob_backward = 0.0
    
    do_mask = fn.random.coin_flip(probability=prob, dtype=types.DALIDataType.BOOL)
    do_backward = fn.random.coin_flip(probability=prob_backward, dtype=types.DALIDataType.BOOL)

    if train:
        seq_backward = sequence[1:-1,:]
        sequence = sequence[1:-1,:]
        masks = masks[1:-1,:]
    else:
        seq_backward = sequence
        sequence = sequence
        masks = masks

    if do_mask:
        masked_sequence = (sequence - 0.5) * masks + 0.5
    else:
        masked_sequence = sequence

    label_backward = fn.zeros(shape=1, dtype=types.DALIDataType.INT64)
    if do_backward:
        seq_backward = seq_backward[::-1,:,:,:]
        label_backward = fn.ones(shape=1, dtype=types.DALIDataType.INT64)

    sequence = fn.transpose(sequence, perm=[3,0,1,2])
    masked_sequence = fn.transpose(masked_sequence, perm=[3,0,1,2])
    seq_backward = fn.transpose(seq_backward, perm=[3,0,1,2])

    return sequence, masked_sequence.gpu(), seq_backward, label_backward.gpu(), key.gpu(), prefix.gpu()

if __name__ == "__main__":
    pipe = simple_pipeline(image_dir, batch_size=max_batch_size, num_threads=1, device_id=0)
    pipe.build()



    

