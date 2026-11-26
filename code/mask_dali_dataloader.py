from nvidia.dali import pipeline_def, pipeline
from nvidia.dali.pipeline import do_not_convert
import nvidia.dali.fn as fn
import nvidia.dali.types as types

from hibrid_mask_dataloader import MaskGeneratorTorch

import csv
import random
import numpy as np
import os
import struct

import mmap

random.seed(12334)

# PID of the process that imported this module. DALI's ``external_source`` with
# ``parallel=True`` forks workers *after* this seed was applied, so every worker
# would inherit the same RNG state and emit identical masks/shuffles.
_seed_worker_origin_pid = os.getpid()
_seed_worker_current_pid = None


def seed_worker_if_forked(base=12334):
    """Re-seed the global RNGs once per forked worker process.

    Only runs when called from a process whose PID differs from the import-time
    PID (i.e. a forked DALI worker); the main process is left untouched so the
    seed set by ``trainUtils.set_seed`` stays intact. Idempotent per process, so
    the stream advances normally across samples instead of resetting every call.
    """
    global _seed_worker_current_pid
    pid = os.getpid()
    if pid == _seed_worker_origin_pid or pid == _seed_worker_current_pid:
        return
    seed = base + (pid % 100000)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    _seed_worker_current_pid = pid


@do_not_convert
class ExternalInputIterator(object):
    def __init__(self, batch_size, csv_file, seq_len=7):
        self.batch_size = batch_size
        self.csv_file = csv_file
        
        # Memory-map the file for fast random access
        self.file = open(csv_file, 'r+b')
        self.mm = mmap.mmap(self.file.fileno(), 0)
        
        # Index line positions
        self.line_positions = [0]
        self.mm.seek(0)
        for line in iter(self.mm.readline, b""):
            self.line_positions.append(self.mm.tell())
        
        # Remove last position (EOF)
        self.line_positions.pop()
        
        self.num_lines = len(self.line_positions)
        self.full_iterations = self.num_lines // self.batch_size
        self.seq_len = seq_len
        self.p = 0.5
        self.indices = [i for i in range(self.seq_len)]
        self.num_pos = 6
    
    def get_row_by_index(self, idx):
        """Read row from memory-mapped file"""
        self.mm.seek(self.line_positions[idx])
        line = self.mm.readline().decode('utf-8').strip()
        reader = csv.reader([line], delimiter=';')
        return next(reader)
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'file'):
            self.file.close()
    
    def generate_positions(self,):
        positions = []
        
        for _ in range(self.num_pos):
            pos = np.random.randint(1, 4)
            positions.append(pos)

        num_left = int(self.num_pos // 2)
        
        left_part = [-1*i for i in positions[:num_left]]
        for i in range(num_left - 1):
            left_part[i] += sum(left_part[i + 1:])

        right_part = positions[num_left:]
        for i in range(1, num_left):
            right_part[i] += right_part[i - 1]

        new_pos = left_part + [0] + right_part
        new_pos = [i+15 for i in new_pos]

        return new_pos
    
    
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
        
        #row = self.files[sample_idx]
        row = self.get_row_by_index(sample_idx)
        num_files = len(row)
        for img_path in row[2:num_files]:
            with open(img_path, "rb") as f:
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
        
        resnet = np.squeeze(np.load(row[0]).astype(np.float32))
        yolo = np.load(row[1]).astype(np.float32)
        
        batch.append(resnet)
        batch.append(yolo)
        batch.append(np.array(self.generate_positions()).astype(np.int32))

        key, prefix = self.get_key_prefix_encode(row[0])
        batch.append(key)
        batch.append(prefix)

        return batch

@do_not_convert
class MaskedInputIterator(object):
    def __init__(self, batch_size, csv_file, seq_len:int=7,
                 input_size:int=64,
                 train:bool=True, 
                 cover_factor:float=0.3,
                 square_size:int=3):
        
        self.batch_size = batch_size
        self.csv_file = csv_file
        self.input_size = input_size
        self.cover_factor = cover_factor
        self.square_size = square_size
        self.train = train
        
        # Memory-map the file for fast random access
        self.file = open(csv_file, 'r+b')
        self.mm = mmap.mmap(self.file.fileno(), 0)
        
        # Index line positions
        self.line_positions = [0]
        self.mm.seek(0)
        for line in iter(self.mm.readline, b""):
            self.line_positions.append(self.mm.tell())
        
        # Remove last position (EOF)
        self.line_positions.pop()
        
        self.num_lines = len(self.line_positions)
        self.full_iterations = self.num_lines // self.batch_size
        self.seq_len = seq_len
        self.p = 0.5

        self.mask_generator = MaskGeneratorTorch(height=input_size, width=input_size,
                                                 percent=cover_factor, square_size=square_size,
                                                 device='cpu')
        
        self.n_squares = self.mask_generator.num_squares
        self.radius = self.square_size // 2
    
    def get_row_by_index(self, idx):
        """Read row from memory-mapped file"""
        self.mm.seek(self.line_positions[idx])
        line = self.mm.readline().decode('utf-8').strip()
        reader = csv.reader([line], delimiter=';')
        return next(reader)
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'file'):
            self.file.close()
    
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
        
        row = self.get_row_by_index(sample_idx)
        num_files = len(row)

        for img_path in row[2:num_files]:
            with open(img_path, "rb") as f:
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
        
        num_mask = len(row[2:num_files])

        for _ in range(num_mask):
            mask = np.ones((self.input_size, self.input_size, 3), dtype=np.uint8)
            if self.train:
                batch_mask_x, batch_mask_y = self.mask_generator.get_squares_coords()
                n_gen_squares = min(len(batch_mask_x), len(batch_mask_y))
                for i in range(n_gen_squares):
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
        central_indexes = types.Constant(np.array([12,13,14,15,16,17,18]), shape=(7,), dtype=types.DALIDataType.INT32)
        #middle_frame = 15
        prob = 0.9
    else:
        central_indexes = types.Constant(np.array([0,1,2,3,4,5,6]), shape=(7,), dtype=types.DALIDataType.INT32)
        #middle_frame = 3
        prob = 0.0
    
    do_mask = fn.random.coin_flip(probability=prob, dtype=types.DALIDataType.BOOL)

    if train:
        sequence = fn.sequence_rearrange(sequence, new_order=central_indexes) #sequence[1:-1,:]
        masks = fn.sequence_rearrange(masks, new_order=central_indexes)
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

@do_not_convert
class MaskedInputIteratorV2(object):
    def __init__(self, batch_size, csv_file, seq_len:int=7,
                 input_size:int=64, train:bool=True, 
                 cover_factor:float=0.3, square_size:int=3):
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.train = train
        self.seq_len = seq_len
        self.square_size = square_size
        self.radius = self.square_size // 2
        
        # 1. Carregando tudo na RAM (Thread-Safe para leitura)
        with open(csv_file, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f.readlines()]
            
        self.num_lines = len(self.lines)
        self.full_iterations = self.num_lines // self.batch_size

        self.mask_generator = MaskGeneratorTorch(height=input_size, width=input_size,
                                                 percent=cover_factor, square_size=square_size,
                                                 device='cpu')
    
    def get_key_prefix_encode(self, resnet_row, is_full_frame=False):
        # Full-frame rows carry the key path in row[0]:
        #   .../val_frames/<sample_id>/<sample_id>_<frame_id>.png
        # so the "<sample_id>_<frame_id>" key is just the file stem. The orig rows
        # encode the key two/three levels up (YOLO-crop .npy path structure).
        if is_full_frame:
            key = os.path.splitext(os.path.basename(resnet_row))[0]
        else:
            dirname = os.path.dirname(os.path.dirname(resnet_row))
            frame_id = os.path.basename(dirname)
            dirname = os.path.dirname(dirname)
            sample_id = os.path.basename(dirname)
            key = f"{sample_id}_{frame_id}"
        key = key.encode('utf-8')
        length_prefix = struct.pack('<i', len(key))
        return np.frombuffer(length_prefix + key, dtype=np.int8), np.frombuffer(length_prefix, dtype=np.int8)

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()
        
        batch = []
        row = next(csv.reader([self.lines[sample_idx]], delimiter=';'))
        
        # Lê os JPEGs
        for img_path in row[2:]:
            with open(img_path, "rb") as f:
                #print(f"File {img_path}")
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
        
        # 2. TUBELET MASKING: Gera a máscara espacial UMA vez
        # Shape: (1, H, W, 1) para permitir broadcasting no DALI
        mask_2d = np.ones((1, self.input_size, self.input_size, 1), dtype=np.float32)
        
        if self.train:
            batch_mask_x, batch_mask_y = self.mask_generator.get_squares_coords()
            n_gen = min(len(batch_mask_x), len(batch_mask_y))
            for i in range(n_gen):
                r, c = batch_mask_x[i], batch_mask_y[i]
                r_start, r_end = max(0, r-self.radius), min(self.input_size, r+self.radius)
                c_start, c_end = max(0, c-self.radius), min(self.input_size, c+self.radius)
                mask_2d[:, r_start:r_end, c_start:c_end, :] = 0.0

        # Adiciona apenas UMA máscara espacial ao batch (DALI vai expandir no tempo)
        batch.append(mask_2d)

        key, prefix = self.get_key_prefix_encode(row[0])
        batch.append(key)
        batch.append(prefix)
        #print(f"Len Batch {len(batch)}")

        return batch

@do_not_convert
class MaskedInputTimeShuffleIteratorV2(MaskedInputIteratorV2):
    def __init__(self, batch_size, csv_file, seq_len:int=9,
                     input_size:int=64, train:bool=True, mask_prob:float=0.5,
                     shuffle_temporal:bool=True, 
                     cover_factor:float=0.3, square_size:int=3):

        if train and seq_len != 9:
            raise ValueError(f"Invalid Sequence Length {seq_len}.")
        
        super().__init__(batch_size=batch_size, csv_file=csv_file, seq_len=seq_len,
                        input_size=input_size, train=train, 
                        cover_factor=cover_factor, square_size=square_size)
        self.middle_frame = self.seq_len // 2
        self.shuffle_temporal = shuffle_temporal
        self.mask_prob = mask_prob

    def gerar_combinacao(self, N: int, start:int=0):
        while True:
            indices = range(start, start + N)
            v = list(indices)
            if not self.shuffle_temporal:
                return v, 0
            
            coin = random.randint(0, 1)
            if coin == 0:
                return v, coin
            
            random.shuffle(v)
            acertos = sum(1 for i,idx in enumerate(indices) if v[i] == idx)
            if acertos <= 1:
                return v, coin

    def generate_positions(self,):
        left_part, left_label = self.gerar_combinacao(N=self.middle_frame, start=0)
        right_part, right_label = self.gerar_combinacao(N=self.middle_frame, start=self.middle_frame+1)
        indices = left_part + [self.middle_frame] + right_part
        label = 2 * left_label + right_label

        return np.array(indices, dtype=np.int32), np.array([label], dtype=np.int32)

    def __call__(self, sample_info):
        seed_worker_if_forked()
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()
        
        batch = []
        row = next(csv.reader([self.lines[sample_idx]], delimiter=';'))
        
        for img_path in row[2:]:
            with open(img_path, "rb") as f:
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
        
        mask_2d = np.ones((1, self.input_size, self.input_size, 1), dtype=np.float32)
        
        if self.mask_prob > 0.0:
            batch_mask_x, batch_mask_y = self.mask_generator.get_squares_coords()
            n_gen = min(len(batch_mask_x), len(batch_mask_y))
            for i in range(n_gen):
                r, c = batch_mask_x[i], batch_mask_y[i]
                r_start, r_end = max(0, r-self.radius), min(self.input_size, r+self.radius)
                c_start, c_end = max(0, c-self.radius), min(self.input_size, c+self.radius)
                mask_2d[:, r_start:r_end, c_start:c_end, :] = 0.0
        
        batch.append(mask_2d) #31

        indices, label = self.generate_positions()
        batch.append(indices) #32
        batch.append(label) #33

        key, prefix = self.get_key_prefix_encode(row[0])
        batch.append(key) #34
        batch.append(prefix) #35

        return batch


@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0)
def masked_pipe_v2(ann_file, num_frames, pipe_batch_size=4, shape=(64,64), seq_len=7, train=True, device='gpu', cover_factor:float=0.3, square_size:int=5):
    num_outputs = num_frames + 3
    *outputs, = fn.external_source(source=MaskedInputIteratorV2(csv_file=ann_file,
                                                              batch_size=pipe_batch_size,
                                                              train=train,
                                                              cover_factor=cover_factor,
                                                              square_size=square_size),
                                   num_outputs=num_outputs,
                                   batch=False)
    
    jpegs = outputs[:num_frames]
    mask_2d = outputs[num_frames] # A máscara espacial única
    key = outputs[-2]
    prefix = outputs[-1]
    
    # Decode na GPU SOTA
    images = fn.decoders.image(jpegs, device="mixed")
    
    # Processamento Espaço-Temporal
    sequence = fn.resize(images, size=shape, device=device)
    sequence = fn.stack(*sequence) # Shape: (T, H, W, 3)
    sequence = fn.reshape(sequence, layout="FHWC")
    sequence = fn.crop_mirror_normalize(sequence, dtype=types.FLOAT, std=[255.0], output_layout="FHWC")

    # Passa a máscara para a GPU
    mask_gpu = mask_2d.gpu()

    #seq_len = 7 # Número de frames que queremos extrair

    if train:
        c_idx = (num_frames // 2) - (seq_len // 2)
        start_idx = types.Constant(np.array([c_idx], dtype=np.int32))
        prob = 0.9
    else:
        max_start = num_frames - seq_len
        if max_start == 0:
            start_idx = 0
        else:
            start_idx = fn.random.uniform(range=[0, max_start], dtype=types.INT32, shape=[1])
        prob = 0.0
    
    window_shape = types.Constant(np.array([seq_len], dtype=np.int32))
    sequence = fn.slice(sequence, start_idx, window_shape, axes=[0])

    # Lógica condicional de aplicação da máscara
    do_mask = fn.random.coin_flip(probability=prob, dtype=types.BOOL)

    if do_mask:
        masked_sequence = (sequence - 0.5) * mask_gpu + 0.5
        final_mask = sequence * 0.0 + mask_gpu
    else:
        masked_sequence = sequence
        final_mask = sequence * 0.0 + 1.0
    
    sequence = fn.transpose(sequence, perm=[3,0,1,2])
    masked_sequence = fn.transpose(masked_sequence, perm=[3,0,1,2])
    final_mask = fn.transpose(final_mask, perm=[3,0,1,2])

    return sequence, masked_sequence, final_mask, key.gpu(), prefix.gpu()

@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0)
def masked_arrow_pipe_v2(ann_file, num_frames, pipe_batch_size=4, shape=(64,64), seq_len=7, train=True, device='gpu', cover_factor:float=0.3, square_size:int=5):
    num_outputs = num_frames + 3
    *outputs, = fn.external_source(source=MaskedInputIteratorV2(csv_file=ann_file,
                                                              batch_size=pipe_batch_size,
                                                              train=train,
                                                              cover_factor=cover_factor,
                                                              square_size=square_size),
                                   num_outputs=num_outputs,
                                   batch=False)
    
    jpegs = outputs[:num_frames]
    mask_2d = outputs[num_frames] # A máscara espacial única
    key = outputs[-2]
    prefix = outputs[-1]
    
    # Decode na GPU SOTA
    images = fn.decoders.image(jpegs, device="mixed")
    
    # Processamento Espaço-Temporal
    sequence = fn.resize(images, size=shape, device=device)
    sequence = fn.stack(*sequence) # Shape: (T, H, W, 3)
    sequence = fn.reshape(sequence, layout="FHWC")
    sequence = fn.crop_mirror_normalize(sequence, dtype=types.FLOAT, std=[255.0], output_layout="FHWC")

    # Passa a máscara para a GPU
    mask_gpu = mask_2d.gpu()

    if train:
        c_idx = (num_frames // 2) - (seq_len // 2)
        start_idx = types.Constant(np.array([c_idx], dtype=np.int32))
        prob = 0.9
        prob_arrow = 0.5
    else:
        max_start = num_frames - seq_len
        if max_start == 0:
            start_idx = 0
        else:
            start_idx = fn.random.uniform(range=[0, max_start], dtype=types.INT32, shape=[1])
        prob = 0.0
        prob_arrow = 0.0
    
    window_shape = types.Constant(np.array([seq_len], dtype=np.int32))
    sequence = fn.slice(sequence, start_idx, window_shape, axes=[0])

    # Lógica condicional de aplicação da máscara
    do_mask = fn.random.coin_flip(probability=prob, dtype=types.BOOL)
    do_arrow = fn.random.coin_flip(probability=prob_arrow, dtype=types.BOOL)

    if do_mask:
        masked_sequence = (sequence - 0.5) * mask_gpu + 0.5
        final_mask = sequence * 0.0 + mask_gpu
    else:
        masked_sequence = sequence
        final_mask = sequence * 0.0 + 1.0
    
    if do_arrow:
        arrow_sequence = sequence[::-1,:,:,:]
        label_backward = fn.ones(shape=1, dtype=types.DALIDataType.INT64)
    else:
        arrow_sequence = sequence
        label_backward = fn.zeros(shape=1, dtype=types.DALIDataType.INT64)
    
    sequence = fn.transpose(sequence, perm=[3,0,1,2])
    arrow_sequence = fn.transpose(arrow_sequence, perm=[3,0,1,2])
    masked_sequence = fn.transpose(masked_sequence, perm=[3,0,1,2])
    final_mask = fn.transpose(final_mask, perm=[3,0,1,2])

    return sequence, masked_sequence, final_mask, arrow_sequence, label_backward.gpu(), key.gpu(), prefix.gpu()

@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0)
def masked_time_pipe(ann_file, num_frames, pipe_batch_size=4, shape=(64,64), seq_len=9,
                     device='gpu', cover_factor:float=0.3, square_size:int=5,
                     train=True,
                     mask_prob=0.9,           # probabilidade de aplicar máscara
                     shuffle_temporal=True,   # se embaralha temporalmente
                     random_window=True,      # se usa janela aleatória
                     augment=False,           # aug de treino (flip + fotométrico), train-only
                     parallel=False):         # external_source(parallel=...) worker pool
    num_outputs = num_frames + 5
    *outputs, = fn.external_source(source=MaskedInputTimeShuffleIteratorV2(csv_file=ann_file,
                                                              batch_size=pipe_batch_size,
                                                              train=train,
                                                              mask_prob=mask_prob,
                                                              shuffle_temporal=shuffle_temporal,
                                                              seq_len=seq_len,
                                                              cover_factor=cover_factor,
                                                              square_size=square_size),
                                   num_outputs=num_outputs,
                                   batch=False,
                                   parallel=parallel)
    
    jpegs = outputs[:num_frames]
    mask_2d = outputs[num_frames] # A máscara espacial única
    key = outputs[-2]
    prefix = outputs[-1]

    time_indices = outputs[num_frames+1]
    time_label = outputs[num_frames+2]
    
    # Decode na GPU SOTA
    images = fn.decoders.image(jpegs, device="mixed")
    
    # Processamento Espaço-Temporal
    sequence = fn.resize(images, size=shape, device=device)
    sequence = fn.stack(*sequence) # Shape: (T, H, W, 3)
    sequence = fn.reshape(sequence, layout="FHWC")

    if augment and train:
        # Temporally-consistent augmentation: one decision per clip, applied on
        # the (T,H,W,C) sequence so all frames stay consistent. The derived
        # masked_sequence / time_sequence inherit it. Train-only.
        do_flip = fn.random.coin_flip(probability=0.5, dtype=types.DALIDataType.INT32)
        sequence = fn.flip(sequence, horizontal=do_flip)
        sequence = fn.color_twist(sequence,
                                  brightness=fn.random.uniform(range=[0.8, 1.2], dtype=types.FLOAT),
                                  contrast=fn.random.uniform(range=[0.8, 1.2], dtype=types.FLOAT),
                                  saturation=fn.random.uniform(range=[0.8, 1.2], dtype=types.FLOAT),
                                  hue=fn.random.uniform(range=[-10.0, 10.0], dtype=types.FLOAT))

    sequence = fn.crop_mirror_normalize(sequence, dtype=types.FLOAT, std=[255.0], output_layout="FHWC")

    # Passa a máscara para a GPU
    mask_gpu = mask_2d.gpu()

    # Seleção da janela
    if random_window:
        max_start = num_frames - seq_len
        if max_start > 0:
            start_idx = fn.random.uniform(range=[0, max_start], dtype=types.INT32, shape=[1])
        else:
            start_idx = types.Constant(np.array([0], dtype=np.int32))
    else:
        c_idx = (num_frames // 2) - (seq_len // 2)
        start_idx = types.Constant(np.array([c_idx], dtype=np.int32))

    # if train:
    #     c_idx = (num_frames // 2) - (seq_len // 2)
    #     start_idx = types.Constant(np.array([c_idx], dtype=np.int32))
    #     prob = 0.9
    # else:
    #     max_start = num_frames - seq_len
    #     if max_start == 0:
    #         start_idx = 0
    #     else:
    #         start_idx = fn.random.uniform(range=[0, max_start], dtype=types.INT32, shape=[1])
    #     prob = 0.0
    
    window_shape = types.Constant(np.array([seq_len], dtype=np.int32))
    sequence = fn.slice(sequence, start_idx, window_shape, axes=[0])

    # Lógica condicional de aplicação da máscara
    do_mask = fn.random.coin_flip(probability=mask_prob, dtype=types.BOOL)

    if do_mask:
        masked_sequence = (sequence - 0.5) * mask_gpu + 0.5
        final_mask = sequence * 0.0 + mask_gpu
    else:
        masked_sequence = sequence
        final_mask = sequence * 0.0 + 1.0

    if shuffle_temporal:
        time_sequence = fn.sequence_rearrange(sequence, new_order=time_indices)
    else:
        time_sequence = sequence
        time_label = fn.zeros(shape=1, dtype=types.DALIDataType.INT64)

    # if train:
    #     time_sequence = fn.sequence_rearrange(sequence, new_order=time_indices)
    # else:
    #     time_sequence = sequence
    #     time_label = fn.zeros(shape=1, dtype=types.DALIDataType.INT64)
    
    sequence = fn.transpose(sequence, perm=[3,0,1,2])
    time_sequence = fn.transpose(time_sequence, perm=[3,0,1,2])
    masked_sequence = fn.transpose(masked_sequence, perm=[3,0,1,2])
    final_mask = fn.transpose(final_mask, perm=[3,0,1,2])

    return sequence, masked_sequence, final_mask, time_sequence, time_label.gpu(), key.gpu(), prefix.gpu()

@do_not_convert
class MaskedInputIteratorV3(object):
    def __init__(self, batch_size, csv_file, seq_len:int=7,
                 input_size:int=64, train:bool=True, 
                 cover_factor:float=0.3, square_size:int=3):
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.train = train
        self.seq_len = seq_len
        self.square_size = square_size
        self.radius = self.square_size // 2
        
        # 1. Carregando tudo na RAM (Thread-Safe para leitura)
        with open(csv_file, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f.readlines()]
            
        self.num_lines = len(self.lines)
        self.full_iterations = self.num_lines // self.batch_size

        self.mask_generator = MaskGeneratorTorch(height=input_size, width=input_size,
                                                 percent=cover_factor, square_size=square_size,
                                                 device='cpu')
    
    def get_key_prefix_encode(self, img_path:str):
        dirname = os.path.dirname(os.path.dirname(img_path))
        sample_id = os.path.basename(dirname)
        basename = os.path.basename(img_path)
        parts = basename.split('.')[0].split('_')
        frame_id = parts[0]
        
        key = f"{sample_id}_{frame_id}".encode('utf-8')
        length_prefix = struct.pack('<i', len(key))
        
        return np.frombuffer(length_prefix + key, dtype=np.int8), np.frombuffer(length_prefix, dtype=np.int8)

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()
        
        batch = []
        image_path = self.lines[sample_idx]

        with open(image_path, "rb") as f:
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
        
        mask_2d = np.ones((1, self.input_size, self.input_size, 1), dtype=np.float32)
        
        if self.train:
            batch_mask_x, batch_mask_y = self.mask_generator.get_squares_coords()
            n_gen = min(len(batch_mask_x), len(batch_mask_y))
            for i in range(n_gen):
                r, c = batch_mask_x[i], batch_mask_y[i]
                mask_2d[:,r-self.radius:r+self.radius, c-self.radius:c+self.radius,:] = 0.0

        # Adiciona apenas UMA máscara espacial ao batch (DALI vai expandir no tempo)
        batch.append(mask_2d)

        key, prefix = self.get_key_prefix_encode(image_path)
        batch.append(key)
        batch.append(prefix)

        return batch

@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0)
def masked_pipe_v3(ann_file, num_frames, pipe_batch_size=4, shape=(64,64), seq_len=7, train=True, device='gpu', cover_factor:float=0.3, square_size:int=5):
    num_outputs = 4
    *outputs, = fn.external_source(source=MaskedInputIteratorV3(csv_file=ann_file,
                                                              batch_size=pipe_batch_size,
                                                              train=train,
                                                              cover_factor=cover_factor,
                                                              square_size=square_size),
                                   num_outputs=num_outputs,
                                   batch=False)
    
    jpegs = outputs[0]
    mask_2d = outputs[1]
    key = outputs[-2]
    prefix = outputs[-1]
    
    # Decode na GPU SOTA
    images = fn.decoders.numpy(jpegs, device="cpu")
    
    images = images.gpu()
    sequence = fn.resize(images, size=shape, device=device)
    sequence = fn.color_space_conversion(sequence, image_type=types.BGR, output_type=types.RGB)
    sequence = fn.reshape(sequence, layout="FHWC")
    sequence = fn.crop_mirror_normalize(sequence, dtype=types.FLOAT, std=[255.0], output_layout="FHWC")

    # Passa a máscara para a GPU
    mask_gpu = mask_2d.gpu()

    # seq_len = 7 # Número de frames que queremos extrair

    if train:
        start_idx = types.Constant(np.array([12], dtype=np.int32))
        prob = 0.9
    else:
        max_start = num_frames - seq_len
        if max_start == 0:
            start_idx = 0
        else:
            start_idx = fn.random.uniform(range=[0, max_start], dtype=types.INT32, shape=[1])
        prob = 0.0
    
    window_shape = types.Constant(np.array([seq_len], dtype=np.int32))
    sequence = fn.slice(sequence, start_idx, window_shape, axes=[0])

    # Lógica condicional de aplicação da máscara
    do_mask = fn.random.coin_flip(probability=prob, dtype=types.BOOL)

    if do_mask:
        masked_sequence = (sequence - 0.5) * mask_gpu + 0.5
        final_mask = sequence * 0.0 + mask_gpu
    else:
        masked_sequence = sequence
        final_mask = sequence * 0.0 + 1.0
    
    sequence = fn.transpose(sequence, perm=[3,0,1,2])
    masked_sequence = fn.transpose(masked_sequence, perm=[3,0,1,2])
    final_mask = fn.transpose(final_mask, perm=[3,0,1,2])

    return sequence, masked_sequence, final_mask, key.gpu(), prefix.gpu()

@do_not_convert
class MaskedInputIteratorV4(object):
    def __init__(self, batch_size, csv_file, seq_len:int=7,
                 input_size:int=64, train:bool=True, 
                 cover_factor:float=0.3, square_size:int=3):
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.train = train
        self.seq_len = seq_len
        self.square_size = square_size
        self.radius = self.square_size // 2
        
        # 1. Carregando tudo na RAM (Thread-Safe para leitura)
        with open(csv_file, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f.readlines()]
            
        self.num_lines = len(self.lines)
        self.full_iterations = self.num_lines // self.batch_size

        self.mask_generator = MaskGeneratorTorch(height=input_size, width=input_size,
                                                 percent=cover_factor, square_size=square_size,
                                                 device='cpu')
    
    def get_key_prefix_encode(self, img_path:str):
        dirname = os.path.dirname(os.path.dirname(img_path))
        sample_id = os.path.basename(dirname)
        basename = os.path.basename(img_path)
        parts = basename.split('.')[0].split('_')
        frame_id = parts[0]
        
        key = f"{sample_id}_{frame_id}".encode('utf-8')
        length_prefix = struct.pack('<i', len(key))
        
        return np.frombuffer(length_prefix + key, dtype=np.int8), np.frombuffer(length_prefix, dtype=np.int8)

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()
        
        batch = []
        image_path = self.lines[sample_idx]

        image_array = np.load(image_path, mmap_mode='r') 
        batch.append(image_array)
        
        mask_2d = np.ones((1, self.input_size, self.input_size, 1), dtype=np.float32)
        
        if self.train:
            batch_mask_x, batch_mask_y = self.mask_generator.get_squares_coords()
            n_gen = min(len(batch_mask_x), len(batch_mask_y))
            for i in range(n_gen):
                r, c = batch_mask_x[i], batch_mask_y[i]
                mask_2d[:,r-self.radius:r+self.radius, c-self.radius:c+self.radius,:] = 0.0

        # Adiciona apenas UMA máscara espacial ao batch (DALI vai expandir no tempo)
        batch.append(mask_2d)

        key, prefix = self.get_key_prefix_encode(image_path)
        batch.append(key)
        batch.append(prefix)

        return batch
    
@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0)
def masked_pipe_v4(ext_source, num_frames, shape=(64,64), train=True, device='gpu'):
    num_outputs = 4
    *outputs, = fn.external_source(source=ext_source,
                                   num_outputs=num_outputs,
                                   batch=False)
    
    images = outputs[0]
    mask_2d = outputs[1]
    key = outputs[-2]
    prefix = outputs[-1]
    
    images = images.gpu()
    sequence = fn.resize(images, size=shape, device=device)
    sequence = fn.color_space_conversion(sequence, image_type=types.BGR, output_type=types.RGB)
    sequence = fn.reshape(sequence, layout="FHWC")
    sequence = fn.crop_mirror_normalize(sequence, dtype=types.FLOAT, std=[255.0], output_layout="FHWC")

    # Passa a máscara para a GPU
    mask_gpu = mask_2d.gpu()

    seq_len = 7 # Número de frames que queremos extrair

    if train:
        start_idx = types.Constant(np.array([12], dtype=np.int32))
        prob = 0.9
    else:
        max_start = num_frames - seq_len
        if max_start == 0:
            start_idx = 0
        else:
            start_idx = fn.random.uniform(range=[0, max_start], dtype=types.INT32, shape=[1])
        prob = 0.0
    
    window_shape = types.Constant(np.array([seq_len], dtype=np.int32))
    sequence = fn.slice(sequence, start_idx, window_shape, axes=[0])

    # Lógica condicional de aplicação da máscara
    do_mask = fn.random.coin_flip(probability=prob, dtype=types.BOOL)

    if do_mask:
        masked_sequence = (sequence - 0.5) * mask_gpu + 0.5
        final_mask = sequence * 0.0 + mask_gpu
    else:
        masked_sequence = sequence
        final_mask = sequence * 0.0 + 1.0
    
    sequence = fn.transpose(sequence, perm=[3,0,1,2])
    masked_sequence = fn.transpose(masked_sequence, perm=[3,0,1,2])
    final_mask = fn.transpose(final_mask, perm=[3,0,1,2])

    return sequence, masked_sequence, final_mask, key.gpu(), prefix.gpu()