"""DALI pipeline for the motion-guided (frame-difference) region selection.

Replaces the YOLO crop preprocessing + random tube mask with full-frame input
and motion-anchored square masks (see ``motion_region_selector.py``).

Data format (full-frame): the CSV is ``;``-delimited. ``row[0]`` is a key path
``.../val_frames/<sample_id>/<sample_id>_<frame_id>.png`` whose file stem is the
``<sample_id>_<frame_id>`` key (never opened); ``row[1]`` is an empty field and
``row[2:]`` are the full-frame JPEG paths.

Live pieces:
  * ``MotionMaskedTimeShuffleIterator`` — external source: selects the 9-frame
    window in Python, reads the raw JPEG bytes, and emits them + the
    temporal-shuffle indices/label + a shared ``do_flip`` flag. It no longer
    decodes frames or computes the saliency/mask in Python (that full-resolution
    PIL decode + NumPy saliency was the ~1 batch/s bottleneck).
  * ``motion_masked_time_pipe`` — decodes/resizes on the GPU, computes the
    frame-difference saliency and the aggregated map on the GPU, samples the
    motion-anchored squares on the CPU via ``fn.python_function`` (only the cheap
    64x64 selection, keeping ``motion_region_selector`` semantics exact), and
    applies the single tube mask over the whole clip (train/test use the same
    ``cover_factor``/``square_size``).

The temporal-shuffle ``time_sequence`` is built as before.
"""
import csv

import numpy as np

from nvidia.dali import pipeline_def
from nvidia.dali import math as dali_math
from nvidia.dali.pipeline import do_not_convert
import nvidia.dali.fn as fn
import nvidia.dali.types as types

import motion_region_selector
from mask_dali_dataloader import MaskedInputTimeShuffleIteratorV2, seed_worker_if_forked


@do_not_convert
class MotionMaskedTimeShuffleIterator(MaskedInputTimeShuffleIteratorV2):
    """External source for full-frame, motion-anchored square masking.

    Subclasses ``MaskedInputTimeShuffleIteratorV2`` to reuse the CSV loading,
    the key encoding and the temporal-shuffle ``generate_positions`` logic. It
    only selects the frame window, reads the raw JPEG bytes, decides the shared
    ``do_flip`` and packages the batch — decoding and masking happen in
    ``motion_masked_time_pipe``.
    """

    def __init__(self, batch_size, csv_file, seq_len=9, input_size=64,
                 train=True, mask_prob=0.9, shuffle_temporal=True,
                 cover_factor=0.9, square_size=8,
                 augment=False, is_full_frame=True):
        super().__init__(batch_size=batch_size, csv_file=csv_file, seq_len=seq_len,
                         input_size=input_size, train=train, mask_prob=mask_prob,
                         shuffle_temporal=shuffle_temporal,
                         cover_factor=cover_factor, square_size=square_size)
        self.augment = augment
        self.is_full_frame = is_full_frame

    def __call__(self, sample_info):
        seed_worker_if_forked()

        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()

        row = next(csv.reader([self.lines[sample_idx]], delimiter=";"))
        frame_paths = row[2:]          # row[0]=key, row[1]="", row[2:]=JPEGs
        n_total = len(frame_paths)
        seq_len = self.seq_len

        # Select the 9-frame window in Python so the motion is estimated on the
        # same frames the model sees (DALI no longer slices the sequence).
        if n_total > seq_len:
            if self.train:
                start = np.random.randint(0, n_total - seq_len + 1)
            else:
                start = (n_total // 2) - (seq_len // 2)
        else:
            start = 0
        window = frame_paths[start:start + seq_len]

        # Read the raw bytes once; DALI decodes them on the GPU.
        raw = []
        for p in window:
            with open(p, "rb") as f:
                raw.append(np.frombuffer(f.read(), dtype=np.uint8))

        # Horizontal flip is decided here and DALI flips the decoded sequence with
        # the same flag *before* the saliency is computed, so the mask stays
        # consistent with the frames the model sees.
        do_flip = int(self.train and self.augment and (np.random.random() < 0.5))

        batch = list(raw)
        indices, label = self.generate_positions()
        batch.append(indices)                                   # (T,) int32
        batch.append(label)                                     # (1,) int32
        batch.append(np.array(do_flip, dtype=np.int32))         # 0-d int32

        key, prefix = self.get_key_prefix_encode(row[0], is_full_frame=self.is_full_frame)
        batch.append(key)
        batch.append(prefix)

        return batch


@do_not_convert
def _make_mask_from_saliency(cover_factor, square_size, min_saliency, mask_prob):
    """Factory for the CPU mask sampler fed by ``fn.python_function``.

    Receives the aggregated saliency map ``(H, W)`` in ``[0, 1]`` and returns the
    single tube mask ``(1, H, W, 1)`` (``1 = visible``). Keeps the exact
    ``motion_region_selector`` sampling semantics; the static-window fallback
    (near-zero saliency) reuses the grid sampler instead of ``MaskGeneratorTorch``
    so the callback stays torch-free.
    """

    def mask_from_saliency(saliency):
        # DALI can hand the aggregated map with singleton dims attached (e.g.
        # (H, W, 1) when color_space_conversion keeps a trailing channel, or
        # (1, H, W) from the batch axis). Collapse them to the 2D (H, W) map the
        # sampler expects — otherwise ``H, W = saliency.shape`` raises
        # "too many values to unpack".
        saliency = np.squeeze(np.asarray(saliency, dtype=np.float32))
        H, W = saliency.shape
        num_squares = int(cover_factor * H * W / (square_size * square_size))
        centers = motion_region_selector.sample_squares_from_map(
            saliency, num_squares, square_size, min_saliency=min_saliency)
        if not centers and mask_prob > 0.0:
            centers = motion_region_selector.random_grid_squares(
                H, W, num_squares, square_size)
        return motion_region_selector.build_tube_mask(H, W, centers, square_size)

    return mask_from_saliency


@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0)
def motion_masked_time_pipe(ann_file, pipe_batch_size=4, shape=(64, 64),
                            seq_len=9, device="gpu", cover_factor=0.9, square_size=8,
                            train=True, mask_prob=0.9, shuffle_temporal=True,
                            augment=False,
                            motion_method="consecutive", motion_aggregate="sum",
                            motion_min_saliency=0.01,
                            motion_sigma=1.0, motion_res=64,
                            parallel=False):
    # seq_len frames + indices + label + do_flip + key + prefix
    num_outputs = seq_len + 5
    *outputs, = fn.external_source(
        source=MotionMaskedTimeShuffleIterator(
            csv_file=ann_file,
            batch_size=pipe_batch_size,
            train=train,
            mask_prob=mask_prob,
            shuffle_temporal=shuffle_temporal,
            seq_len=seq_len,
            input_size=motion_res,
            cover_factor=cover_factor,
            square_size=square_size,
            augment=augment,
            is_full_frame=True),
        num_outputs=num_outputs,
        batch=False,
        parallel=parallel)

    jpegs = outputs[:seq_len]
    time_indices = outputs[seq_len]
    time_label = outputs[seq_len + 1]
    do_flip = outputs[seq_len + 2]
    key = outputs[-2]
    prefix = outputs[-1]

    images = fn.decoders.image(jpegs, device="mixed")
    sequence = fn.resize(images, size=shape, device=device)
    sequence = fn.stack(*sequence)
    sequence = fn.reshape(sequence, layout="FHWC")

    if augment and train:
        # Flip is shared with the iterator; do it *before* saliency so the mask
        # is computed on the same frames the model sees.
        sequence = fn.flip(sequence, horizontal=do_flip)

    # ---- Saliency map on the GPU (frame difference) ----
    # BT.601 luma, same as motion_region_selector.rgb_to_gray (scale-invariant).
    gray = fn.color_space_conversion(sequence, image_type=types.RGB,
                                     output_type=types.GRAY, device=device)
    gray = fn.cast(gray, dtype=types.FLOAT)

    if motion_method == "consecutive":
        prev = fn.slice(gray, start=[-1], shape=[seq_len], axes=[0],
                        out_of_bounds_policy="edge")
        nxt = fn.slice(gray, start=[1], shape=[seq_len], axes=[0],
                       out_of_bounds_policy="edge")
        sal = 0.5 * (dali_math.abs(gray - prev) + dali_math.abs(gray - nxt))
    elif motion_method == "central":
        center = fn.slice(gray, start=[seq_len // 2], shape=[1], axes=[0])
        sal = dali_math.abs(gray - center)
    else:
        raise ValueError(f"motion_method {motion_method!r} is not supported by the "
                         "GPU path; use 'consecutive' or 'central'")

    sal = fn.gaussian_blur(sal, sigma=[motion_sigma, motion_sigma], device=device)
    sal = sal / (fn.reductions.max(sal, keep_dims=True) + 1e-8)

    if motion_aggregate == "sum":
        agg = fn.reductions.sum(sal, axes=[0])
    elif motion_aggregate == "mean":
        agg = fn.reductions.mean(sal, axes=[0])
    elif motion_aggregate == "maximum":
        agg = fn.reductions.max(sal, axes=[0])
    else:
        raise ValueError(f"motion_aggregate {motion_aggregate!r} not supported")

    agg = agg / (fn.reductions.max(agg, keep_dims=True) + 1e-8)

    # ---- Mask generation on the CPU (exact motion_region_selector sampling) ----
    mask = fn.python_function(
        fn.copy(agg, device="cpu"),
        function=_make_mask_from_saliency(cover_factor, square_size,
                                          motion_min_saliency, mask_prob),
        num_outputs=1, device="cpu")
    mask_gpu = mask.gpu()

    if augment and train:
        # Photometric jitter *after* the saliency branch (saliency is computed on
        # un-jittered frames, matching the old external-source behaviour).
        sequence = fn.color_twist(
            sequence,
            brightness=fn.random.uniform(range=[0.8, 1.2], dtype=types.FLOAT),
            contrast=fn.random.uniform(range=[0.8, 1.2], dtype=types.FLOAT),
            saturation=fn.random.uniform(range=[0.8, 1.2], dtype=types.FLOAT),
            hue=fn.random.uniform(range=[-10.0, 10.0], dtype=types.FLOAT))

    sequence = fn.crop_mirror_normalize(sequence, dtype=types.FLOAT, std=[255.0],
                                        output_layout="FHWC")

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

    sequence = fn.transpose(sequence, perm=[3, 0, 1, 2])
    time_sequence = fn.transpose(time_sequence, perm=[3, 0, 1, 2])
    masked_sequence = fn.transpose(masked_sequence, perm=[3, 0, 1, 2])
    final_mask = fn.transpose(final_mask, perm=[3, 0, 1, 2])

    return (sequence, masked_sequence, final_mask, time_sequence,
            time_label.gpu(), key.gpu(), prefix.gpu())
