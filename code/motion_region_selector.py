"""Pure-NumPy motion-based region selection (no DALI, no torch).

Replaces the YOLO object-crop preprocessing + random tube mask with
frame-difference saliency: in a window of 9 consecutive *full* frames we compute
a per-frame motion map, aggregate it into a single saliency map, and sample
square mask centres from its high-saliency regions. The resulting single tube
mask (shared across the window) is the set of "candidate regions" that become
the holes of the reconstruction proxy task.

This module is shared by:

  * ``MotionMaskedTimeShuffleIterator`` (the DALI external source in
    ``mask_dali_dataloader.py``), and
  * ``visualize_motion_regions.py`` (standalone inspection, no DALI),

so the inspection output is exactly what the training pipeline feeds the model.

Frame-difference logic (per window of ``T`` frames ``G_0..G_{T-1}``, luminance):

  consecutive :  M_t = |G_t - G_{t-1}| + |G_t - G_{t+1}|   (edges use one side)
  median      :  M_t = |G_t - median(G_0..G_{T-1})|        (robust to slow motion)
  both        :  M_t = mean(consecutive, median)
  central     :  M_t = |G_t - G_{T//2}|                    (central frame as reference)

``M`` is smoothed and normalised by the window max, then the per-frame maps are
aggregated into a single map (``sum``/``mean``/``maximum``) and square centres
are sampled from its high-saliency regions. Regions with near-zero aggregated
saliency get no squares, and the resulting single mask is shared across the whole
window (like the original random tube mask). The mask convention matches the rest
of the repo: ``1 = visible``, ``0 = hole``.

The ``central`` method also pairs with a different, standalone mask generator
(``central_saliency_mask``): instead of squares, it turns the highest-saliency
pixels into holes (see below). It is **not** wired into the DALI pipeline.
"""
import numpy as np

# BT.601 luma weights (same as most grayscale conversions).
_RGB_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Default fallback when no rng is supplied: the module-level numpy RNG, which
# ``trainUtils.set_seed`` seeds to 12334 (and which advances across epochs, so
# masks vary epoch to epoch like the current random-square sampler).
_DEFAULT_RNG = np.random


def rgb_to_gray(frames):
    """``frames``: (T, H, W, 3) uint8 -> (T, H, W) float32 in [0, 1]."""
    f = np.asarray(frames, dtype=np.float32) / 255.0
    return (f[..., 0] * _RGB_WEIGHTS[0]
            + f[..., 1] * _RGB_WEIGHTS[1]
            + f[..., 2] * _RGB_WEIGHTS[2])


def _gaussian_kernel(sigma):
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def gaussian_blur(x, sigma):
    """Separable Gaussian blur over the last two axes of ``x``.

    Pure NumPy (reflect padding + a vectorised separable convolution via
    ``sliding_window_view``), so the module has no scipy dependency. ``sigma <= 0``
    is a no-op. The previous ``np.apply_along_axis`` + ``np.convolve`` version was
    ~7x slower and held the GIL (it was a hot spot in the DALI external source).
    """
    x = np.asarray(x, dtype=np.float32)
    if sigma is None or sigma <= 0:
        return x
    k = _gaussian_kernel(sigma)
    pad = len(k) // 2

    def _blur_axis(arr, axis):
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (pad, pad)
        padded = np.pad(arr, pad_width, mode="reflect")
        window = np.lib.stride_tricks.sliding_window_view(padded, len(k), axis=axis)
        return np.einsum("...k,k->...", window, k).astype(np.float32)

    return _blur_axis(_blur_axis(x, -1), -2).astype(np.float32)


def compute_motion_saliency(gray, method="consecutive", sigma=1.0):
    """``gray``: (T, H, W) float32 in [0, 1] -> (T, H, W) float32 in [0, 1].

    Methods:
      consecutive : M_t = |G_t - G_{t-1}| + |G_t - G_{t+1}|  (edges use one side)
      median      : M_t = |G_t - median(G_0..G_{T-1})|
      both        : M_t = mean(consecutive, median)
      central     : M_t = |G_t - G_{T//2}|  (central frame as reference; the
                    central frame itself has ~0 saliency)

    Returns a per-frame saliency map normalised by the window max (at least one
    frame reaches 1.0; a fully static window stays ~0).
    """
    gray = np.asarray(gray, dtype=np.float32)
    T = gray.shape[0]

    if method == "central":
        center = gray[T // 2]
        sal = np.abs(gray - center)                       # (T,H,W) broadcast
    else:
        if method not in ("consecutive", "median", "both"):
            raise ValueError(f"Invalid method {method!r}; "
                             "use 'consecutive', 'median', 'both' or 'central'")
        use_cons = method in ("consecutive", "both")
        use_med = method in ("median", "both")
        bg = np.median(gray, axis=0) if use_med else None

        sal = np.zeros_like(gray)
        for t in range(T):
            acc = []
            if use_cons:
                if t > 0:
                    acc.append(np.abs(gray[t] - gray[t - 1]))
                if t < T - 1:
                    acc.append(np.abs(gray[t] - gray[t + 1]))
            if use_med:
                acc.append(np.abs(gray[t] - bg))
            if acc:
                sal[t] = np.mean(np.stack(acc, axis=0), axis=0)

    sal = gaussian_blur(sal, sigma)
    mx = float(sal.max())
    if mx > 1e-8:
        sal = sal / mx
    return sal


def aggregate_saliency(saliency, mode="sum"):
    """Aggregate per-frame saliency (T, H, W) into a single (H, W) map.

    ``mode``:
      sum      : S = sum_t M_t     (accumulated saliency across the window)
      mean     : S = mean_t M_t    (same ranking as ``sum``, scaled by 1/T)
      maximum  : S = max_t M_t     (peak saliency across the window)

    The result is normalised to [0, 1] so downstream "almost zero" thresholds are
    scale-invariant. Note that ``mean`` and ``sum`` differ only by the constant
    1/T, so after normalisation they coincide; ``maximum`` is distinct.
    """
    saliency = np.asarray(saliency, dtype=np.float32)
    if mode == "sum":
        out = saliency.sum(axis=0)
    elif mode == "mean":
        out = saliency.mean(axis=0)
    elif mode == "maximum":
        out = saliency.max(axis=0)
    else:
        raise ValueError(f"Invalid aggregation {mode!r}; "
                         "use 'sum', 'mean' or 'maximum'")
    mx = float(out.max())
    if mx > 1e-8:
        out = out / mx
    return out.astype(np.float32)


def sample_squares_from_map(saliency_map, num_squares, square_size,
                            min_saliency=0.01, rng=None):
    """Sample square centres from a single saliency map, high-density only.

    Only pixels whose saliency exceeds ``min_saliency`` (a fraction of the map
    max) are eligible, so near-zero / static regions get **no** squares. Among
    the eligible pixels, centres are sampled without replacement weighted by the
    saliency, so higher-density regions receive more squares. If fewer than
    ``num_squares`` pixels are eligible, all of them are used.

    ``saliency_map``: (H, W) float32 in [0, 1] -> list of ``(r, c)`` centres.
    """
    rng = rng if rng is not None else _DEFAULT_RNG
    saliency_map = np.asarray(saliency_map, dtype=np.float32)
    H, W = saliency_map.shape

    radius = square_size // 2
    r0 = min(radius, H - 1)
    r1 = max(H - radius, r0 + 1)
    c0 = min(radius, W - 1)
    c1 = max(W - radius, c0 + 1)

    valid = saliency_map[r0:r1, c0:c1]          # (h, w)
    eligible = np.flatnonzero(valid > min_saliency)
    if eligible.size == 0:
        return []

    num = int(num_squares)
    n_pick = min(num, eligible.size)
    if n_pick <= 0:
        return []

    w = valid.shape[1]
    p = valid.reshape(-1)[eligible].astype(np.float64)
    p = p / p.sum()
    chosen = rng.choice(eligible, size=n_pick, replace=False, p=p)
    rs = (chosen // w) + r0
    cs = (chosen % w) + c0
    return list(zip(rs.tolist(), cs.tolist()))


def build_tube_mask(H, W, centers, square_size):
    """Build a single tube mask shared across all frames.

    Returns ``(1, H, W, 1)`` float32 with ``1 = visible``, ``0 = hole`` (holes
    are ``square_size x square_size`` blocks around each centre, clipped to the
    image, using the same ``[r - radius, r + radius)`` convention as
    ``MaskedInputIteratorV2``). The pipeline broadcasts this over the time axis.
    """
    mask = np.ones((1, H, W, 1), dtype=np.float32)
    radius = square_size // 2
    for (r, c) in centers:
        rs = max(0, r - radius)
        re = min(H, r + radius)
        cs = max(0, c - radius)
        ce = min(W, c + radius)
        mask[0, rs:re, cs:ce, 0] = 0.0
    return mask


def random_grid_squares(H, W, num_squares, square_size, rng=None):
    """Sample ``(r, c)`` square centres from a fixed stride grid (BEiT-style).

    Pure-NumPy mirror of ``MaskGeneratorTorch.get_squares_coords``: tiles are
    ``2 * radius`` wide on a fixed lattice, and a random subset of them is
    selected (``min(num_squares, n_tiles)`` centres), so the realised coverage
    matches ``cover_factor``. This is the fallback used when a window has
    near-zero saliency (and therefore no motion-anchored squares) inside DALI's
    ``fn.python_function`` callback, which must stay torch-free.
    """
    rng = rng if rng is not None else _DEFAULT_RNG
    radius = square_size // 2
    tile = 2 * radius
    if tile <= 0 or num_squares <= 0:
        return []
    rows = np.arange(0, H - tile + 1, tile)
    cols = np.arange(0, W - tile + 1, tile)
    if rows.size == 0 or cols.size == 0:
        return []
    rr, cc = np.meshgrid(rows + radius, cols + radius, indexing="ij")
    centers_r = rr.reshape(-1)
    centers_c = cc.reshape(-1)
    n_pick = min(int(num_squares), centers_r.size)
    if n_pick <= 0:
        return []
    idx = rng.choice(centers_r.size, size=n_pick, replace=False)
    return list(zip(centers_r[idx].tolist(), centers_c[idx].tolist()))


def motion_mask_from_frames(frames_uint8, square_size=8, cover_factor=0.9,
                            method="consecutive", sigma=1.0, rng=None,
                            aggregate="sum", min_saliency=0.01):
    """Full frames -> a single tube mask anchored on the aggregated saliency.

    Per-frame saliency is aggregated into one map (``aggregate`` = sum/mean/max),
    then squares are placed only in the high-saliency regions of that map, and
    the resulting single mask is shared across the whole T-frame volume.

    ``frames_uint8``: (T, H, W, 3) uint8.

    Returns:
        saliency: (H, W) float32 in [0, 1]   (aggregated)
        mask:     (1, H, W, 1) float32, 1 = visible, 0 = hole
        centers:  list of ``(r, c)`` square centres
    """
    rng = rng if rng is not None else _DEFAULT_RNG
    frames_uint8 = np.asarray(frames_uint8, dtype=np.uint8)
    T, H, W, _ = frames_uint8.shape

    num_squares = int(cover_factor * H * W / (square_size * square_size))
    centers = np.array([], dtype=np.int32)

    gray = rgb_to_gray(frames_uint8)
    per_frame = compute_motion_saliency(gray, method=method, sigma=sigma)
    saliency = aggregate_saliency(per_frame, mode=aggregate)
    centers = sample_squares_from_map(saliency, num_squares, square_size,
                                        min_saliency=min_saliency, rng=rng)
    mask = build_tube_mask(H, W, centers, square_size)
    return saliency, mask, centers


def _dilate_binary(binary, radius):
    """Binary dilation with a square structuring element (no scipy).

    ``binary``: (T, H, W) -> dilated (T, H, W) in {0, 1}. ``radius <= 0`` is a
    no-op. Small images only, so the naive shifted-max is fine.
    """
    binary = np.asarray(binary, dtype=np.float32)
    if radius <= 0:
        return binary
    T, H, W = binary.shape
    pad = np.pad(binary, ((0, 0), (radius, radius), (radius, radius)),
                 mode="constant", constant_values=0.0)
    out = np.zeros_like(binary)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            out = np.maximum(
                out,
                pad[:, radius + dr:radius + dr + H, radius + dc:radius + dc + W])
    return out


def saliency_topk_mask(saliency, cover_factor=0.9, dilate=0):
    """Convert a saliency map into a mask whose holes are its highest values.

    For each frame independently, the ``cover_factor`` fraction of pixels with
    the **highest** saliency become holes (0) and the rest stay visible (1), so
    most masked pixels sit on the regions of higher saliency. Each frame gets
    exactly ``cover_factor`` coverage (ties at the cut are broken arbitrarily by
    ``argpartition``). ``dilate > 0`` spreads the holes a few pixels around the
    peaks (coverage then exceeds ``cover_factor``).

    ``saliency``: (T, H, W) float32 -> ``mask``: (T, H, W, 1) float32.
    """
    saliency = np.asarray(saliency, dtype=np.float32)
    T, H, W = saliency.shape
    n = H * W
    k = int(round(cover_factor * n))
    k = min(max(k, 0), n)

    hole = np.zeros((T, H, W), dtype=np.float32)          # 1 = hole
    if k >= n:
        hole[...] = 1.0
    elif k > 0:
        for t in range(T):
            flat = saliency[t].reshape(-1)
            idx = np.argpartition(flat, n - k)[n - k:]    # k highest-saliency px
            hole[t].reshape(-1)[idx] = 1.0

    if dilate > 0:
        hole = _dilate_binary(hole, dilate)

    return (1.0 - hole)[..., None]                        # 1 = visible, 0 = hole


def central_saliency_mask(frames_uint8, cover_factor=0.9, sigma=1.0, dilate=0):
    """Central-frame-reference saliency + high-saliency (top-k) mask.

    The central frame of the window is the reference: saliency for each frame is
    ``|G_t - G_central|``. The mask is then built by turning the
    highest-``cover_factor``-fraction of saliency pixels into holes, so most
    masked pixels land on the regions that changed most relative to the centre.

    This is a standalone method: it is **not** wired into
    ``motion_mask_dataloader.py`` (which still uses square sampling via
    ``motion_mask_from_frames``).

    ``frames_uint8``: (T, H, W, 3) uint8.

    Returns:
        saliency: (T, H, W) float32 in [0, 1]
        mask:     (T, H, W, 1) float32, 1 = visible, 0 = hole
    """
    gray = rgb_to_gray(frames_uint8)
    saliency = compute_motion_saliency(gray, method="central", sigma=sigma)
    saliency = aggregate_saliency(saliency, mode="sum")
    saliency = np.expand_dims(saliency, axis=0)
    mask = saliency_topk_mask(saliency, cover_factor=cover_factor, dilate=dilate)
    saliency = np.squeeze(saliency)
    return saliency, mask
