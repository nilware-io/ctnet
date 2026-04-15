import torch
import math


def get_1d_dct_matrix(N: int) -> torch.Tensor:
    """
    Generates a 1D Discrete Cosine Transform (Type-II) matrix of size NxN.
    Vectorized implementation — fast even for large N (e.g. 2048).
    """
    k = torch.arange(N, dtype=torch.float64).unsqueeze(1)  # (N, 1)
    n = torch.arange(N, dtype=torch.float64).unsqueeze(0)  # (1, N)
    C = torch.cos(math.pi * (2 * n + 1) * k / (2 * N)) * math.sqrt(2 / N)
    C[0, :] = math.sqrt(1 / N)
    return C.float()


# Global cache for DCT matrices to avoid recomputation and per-layer storage
_dct_matrix_cache: dict[tuple, torch.Tensor] = {}


def get_dct_matrix(N: int, device=None, dtype=None) -> torch.Tensor:
    """
    Get a cached DCT matrix of size NxN on the given device/dtype.
    Creates it on first access, reuses thereafter.
    """
    key = (N, device, dtype)
    if key not in _dct_matrix_cache:
        C = get_1d_dct_matrix(N)
        if device is not None or dtype is not None:
            C = C.to(device=device, dtype=dtype)
        _dct_matrix_cache[key] = C
    return _dct_matrix_cache[key]


def _build_zigzag_weight(K_h: int, K_w: int) -> torch.Tensor:
    """
    Build a weight map based on H.265-style diagonal zig-zag scan order.
    For small spatial kernels (3x3, 7x7).

    Returns:
        Tensor of shape (K_h, K_w) with values in [1.0, K_h*K_w].
    """
    order = torch.zeros(K_h, K_w)
    idx = 0
    for s in range(K_h + K_w - 1):
        if s % 2 == 0:
            for i in range(min(s, K_h - 1), max(s - K_w + 1, 0) - 1, -1):
                j = s - i
                order[i, j] = idx
                idx += 1
        else:
            for i in range(max(s - K_w + 1, 0), min(s, K_h - 1) + 1):
                j = s - i
                order[i, j] = idx
                idx += 1
    total = K_h * K_w
    weight = 1.0 + order * ((total - 1.0) / max(total - 1, 1))
    return weight


def _build_channel_freq_weight(H: int, W: int) -> torch.Tensor:
    """
    Radial frequency weighting for channel-wise DCT coefficients.
    Weight is proportional to normalized frequency distance from DC.

    Unlike zigzag (which scales linearly to H*W and explodes for large
    matrices), this stays in [1.0, 3.0] regardless of size.

    Returns:
        Tensor of shape (H, W) with values in [1.0, 3.0].
    """
    u = torch.arange(H, dtype=torch.float32).unsqueeze(1)  # (H, 1)
    v = torch.arange(W, dtype=torch.float32).unsqueeze(0)  # (1, W)
    freq_dist = (u / max(H - 1, 1)) + (v / max(W - 1, 1))  # 0 to 2
    weight = 1.0 + freq_dist  # DC=1.0, highest freq=3.0
    return weight


def calculate_hevc_rate_proxy(
    weight_dct: torch.Tensor,
    qstep: float = 1.0,
    steepness: float = 10.0,
) -> torch.Tensor:
    """
    Differentiable proxy for H.265 bitrate cost of DCT coefficients.

    Handles both spatial DCT (4D: out_ch, in_ch, K_h, K_w) and
    channel-wise DCT (2D: out_ch, in_ch).

    Models three components of HEVC entropy coding:
      1. Significance cost: ~1 bit per non-zero coefficient
      2. Level cost: log2(1 + |level|) bits per non-zero coefficient
      3. Frequency cost: higher-frequency coefficients cost more

    Returns:
        Scalar estimated rate (in approximate bit-like units).
    """
    if weight_dct.dim() == 4:
        # Spatial DCT: use zigzag scan weights on kernel dims
        K_h, K_w = weight_dct.shape[-2], weight_dct.shape[-1]
        freq_weight = _build_zigzag_weight(K_h, K_w)
    elif weight_dct.dim() == 2:
        # Channel-wise DCT: use radial frequency weights
        H, W = weight_dct.shape
        freq_weight = _build_channel_freq_weight(H, W)
    else:
        raise ValueError(f"Expected 2D or 4D tensor, got {weight_dct.dim()}D")

    freq_weight = freq_weight.to(device=weight_dct.device, dtype=weight_dct.dtype)

    level = weight_dct.abs() / qstep
    significance = torch.sigmoid(steepness * (level - 0.5))
    level_cost = torch.log2(1.0 + level)
    per_coeff_rate = significance * (1.0 + level_cost) * freq_weight

    return per_coeff_rate.sum()


def calculate_hevc_rate_proxy_smooth(
    weight_dct: torch.Tensor,
    qstep: float = 1.0,
    steepness: float = 10.0,
    smooth_weight: float = 0.5,
) -> torch.Tensor:
    """
    H.265 rate proxy that also rewards spatial smoothness.

    In addition to the per-coefficient sparsity cost, this penalizes
    differences between adjacent coefficients. H.265's CABAC context
    model and inter-pixel prediction compress smooth regions much better
    than noisy ones, even when both are non-sparse.

    The smoothness term models H.265's residual coding: after prediction,
    the residual is smaller for smooth regions → fewer bits.

    Args:
        weight_dct: DCT coefficients (2D or 4D)
        qstep: quantization step
        steepness: sigmoid steepness for significance
        smooth_weight: weight for smoothness term relative to sparsity term
                       (0 = pure sparsity like original, 1 = equal weighting)

    Returns:
        Scalar estimated rate.
    """
    # Sparsity term (same as original)
    sparsity_rate = calculate_hevc_rate_proxy(weight_dct, qstep, steepness)

    if smooth_weight <= 0:
        return sparsity_rate

    # Smoothness term: penalize differences between adjacent coefficients
    # This approximates the residual after H.265's spatial prediction
    # Smaller differences → smaller residuals → fewer bits
    if weight_dct.dim() == 2:
        # Horizontal differences
        h_diff = (weight_dct[:, 1:] - weight_dct[:, :-1]).abs() / qstep
        h_cost = torch.log2(1.0 + h_diff).sum()
        # Vertical differences
        v_diff = (weight_dct[1:, :] - weight_dct[:-1, :]).abs() / qstep
        v_cost = torch.log2(1.0 + v_diff).sum()
        smoothness_cost = h_cost + v_cost
    elif weight_dct.dim() == 4:
        # For spatial DCT: differences along kernel dims
        h_diff = (weight_dct[..., 1:] - weight_dct[..., :-1]).abs() / qstep
        h_cost = torch.log2(1.0 + h_diff).sum()
        v_diff = (weight_dct[..., 1:, :] - weight_dct[..., :-1, :]).abs() / qstep
        v_cost = torch.log2(1.0 + v_diff).sum()
        smoothness_cost = h_cost + v_cost
    else:
        smoothness_cost = torch.tensor(0.0, device=weight_dct.device)

    return sparsity_rate + smooth_weight * smoothness_cost


@torch.no_grad()
def _estimate_2d_image(img: torch.Tensor, bit_depths, results):
    """Estimate H.265 size for a single 2D image (used by estimate_h265_size_bits)."""
    max_tile = 128
    min_dim = 8
    h, w_dim = img.shape

    if h <= max_tile and w_dim <= max_tile:
        tiles = [img]
    else:
        tiles = []
        n_tr = (h + max_tile - 1) // max_tile
        n_tc = (w_dim + max_tile - 1) // max_tile
        th = (h + n_tr - 1) // n_tr
        tw = (w_dim + n_tc - 1) // n_tc
        th = max(min_dim, th + (th % 2))
        tw = max(min_dim, tw + (tw % 2))
        for r in range(0, h, th):
            for c in range(0, w_dim, tw):
                tile = img[r:min(r + th, h), c:min(c + tw, w_dim)]
                tiles.append(tile)

    for tile in tiles:
        n_pixels = tile.numel()
        if n_pixels == 0:
            continue

        fmin = tile.min().item()
        fmax = tile.max().item()
        span = fmax - fmin

        for b in bit_depths:
            max_val = (1 << b) - 1
            if span == 0:
                results[b] += n_pixels * 0.1
                continue

            norm_factor = max_val / span
            center = (fmin + fmax) / 2.0
            half = max_val / 2.0

            pixels = ((tile.float() - center) * norm_factor + half).round()
            pixels = pixels.clamp(0, max_val).long()

            counts = torch.bincount(pixels.flatten(), minlength=max_val + 1)
            probs = counts.float() / n_pixels
            probs = probs[probs > 0]
            entropy = -(probs * probs.log2()).sum().item()

            h265_efficiency = 1.15
            est_bits = n_pixels * entropy * h265_efficiency
            est_bits += 16 * 8  # per-tile metadata
            results[b] += est_bits


@torch.no_grad()
def estimate_h265_size_bits(model_modules, bit_depths=(8, 10, 12)):
    """
    Estimate H.265 compressed size for all compressible layers at multiple
    bit depths. Handles DCTConv2d, ChannelDCTConv1x1, BatchNorm2d, and Linear.
    """
    import torch.nn as nn
    from dct_layers import DCTConv2d, ChannelDCTConv1x1

    results = {b: 0.0 for b in bit_depths}
    raw_bytes = 0

    for name, m in model_modules:
        if isinstance(m, DCTConv2d):
            w = m.weight_dct.data
            out_ch, in_ch, K_h, K_w = w.shape
            raw_bytes += w.numel() * 4
            img = w.permute(0, 2, 1, 3).reshape(out_ch * K_h, in_ch * K_w)
            _estimate_2d_image(img, bit_depths, results)

        elif isinstance(m, ChannelDCTConv1x1):
            w = m.weight_dct.data
            raw_bytes += w.numel() * 4
            _estimate_2d_image(w, bit_depths, results)

        elif isinstance(m, nn.BatchNorm2d):
            # Stack [weight, bias, running_mean, running_var] as [4, N]
            tensors = [m.weight.data, m.bias.data, m.running_mean, m.running_var]
            img = torch.stack(tensors, dim=0)  # [4, features]
            raw_bytes += img.numel() * 4
            _estimate_2d_image(img, bit_depths, results)

        elif isinstance(m, nn.Linear):
            w = m.weight.data
            raw_bytes += w.numel() * 4
            _estimate_2d_image(w, bit_depths, results)
            if m.bias is not None:
                raw_bytes += m.bias.numel() * 4
                bias_img = m.bias.data.unsqueeze(0)  # [1, out]
                _estimate_2d_image(bias_img, bit_depths, results)

    return results, raw_bytes
