import torch
import math


def get_1d_dct_matrix(N: int) -> torch.Tensor:
    """
    Generates a 1D Discrete Cosine Transform (Type-II) matrix of size NxN.
    Fully differentiable DCT/IDCT via matmul: DCT(x) = C @ x, IDCT(x) = C^T @ x.
    """
    C = torch.zeros((N, N))
    C[0, :] = math.sqrt(1 / N)
    for k in range(1, N):
        for n in range(N):
            C[k, n] = math.sqrt(2 / N) * math.cos(
                math.pi * (2 * n + 1) * k / (2 * N)
            )
    return C


def _build_zigzag_weight(K_h: int, K_w: int) -> torch.Tensor:
    """
    Build a weight map based on H.265-style diagonal zig-zag scan order.

    Coefficients visited later in the scan (higher frequency) get higher
    weights, modeling the fact that CABAC context makes late-scan non-zero
    coefficients more expensive to signal.

    Returns:
        Tensor of shape (K_h, K_w) with values in [1.0, ...] where DC=1.0
        and the last zig-zag position = K_h * K_w.
    """
    order = torch.zeros(K_h, K_w)
    idx = 0
    for s in range(K_h + K_w - 1):
        if s % 2 == 0:
            # upward diagonal
            for i in range(min(s, K_h - 1), max(s - K_w + 1, 0) - 1, -1):
                j = s - i
                order[i, j] = idx
                idx += 1
        else:
            # downward diagonal
            for i in range(max(s - K_w + 1, 0), min(s, K_h - 1) + 1):
                j = s - i
                order[i, j] = idx
                idx += 1
    # Normalize: DC position gets weight 1.0, last position gets weight N
    total = K_h * K_w
    weight = 1.0 + order * ((total - 1.0) / max(total - 1, 1))
    return weight


def calculate_hevc_rate_proxy(
    weight_dct: torch.Tensor,
    qstep: float = 1.0,
    steepness: float = 10.0,
) -> torch.Tensor:
    """
    Differentiable proxy for H.265 bitrate cost of a DCT coefficient block.

    Models three components of HEVC entropy coding:
      1. Significance cost: ~1 bit per non-zero coefficient, approximated by
         a sigmoid soft-threshold on the quantized magnitude.
      2. Level cost: log2(1 + |level|) bits per non-zero coefficient to
         encode the magnitude.
      3. Scan-order cost: coefficients later in the zig-zag scan are more
         expensive due to CABAC context modeling.

    The total rate estimate for one coefficient at position (i,j) is:
        sig(i,j) * (1 + log2(1 + |level(i,j)|)) * scan_weight(i,j)
    where sig is a soft significance indicator and level = |coeff| / qstep.

    Args:
        weight_dct: DCT coefficients, shape (out_ch, in_ch, K_h, K_w).
        qstep: Quantization step size. Larger = coarser quantization = more
               sparsity. Analogous to H.265 QP-derived step size.
        steepness: Controls the sharpness of the soft significance sigmoid.
                   Higher = closer to hard threshold.

    Returns:
        Scalar estimated rate (in approximate bit-like units).
    """
    K_h = weight_dct.shape[-2]
    K_w = weight_dct.shape[-1]

    # Zig-zag scan order weight: DC is cheapest, high-freq is most expensive
    scan_weight = _build_zigzag_weight(K_h, K_w).to(
        device=weight_dct.device, dtype=weight_dct.dtype
    )

    # Quantized magnitude (soft): |coeff| / qstep
    level = weight_dct.abs() / qstep

    # Soft significance: probability that this coefficient survives quantization
    # sigmoid(steepness * (level - 0.5)) ≈ 1 if level > 0.5, ≈ 0 if level < 0.5
    # The 0.5 threshold matches rounding: values < 0.5*qstep round to zero
    significance = torch.sigmoid(steepness * (level - 0.5))

    # Level coding cost: log2(1 + |level|) bits for the magnitude
    level_cost = torch.log2(1.0 + level)

    # Per-coefficient rate: significance flag + level bits, weighted by scan position
    per_coeff_rate = significance * (1.0 + level_cost) * scan_weight

    return per_coeff_rate.sum()


@torch.no_grad()
def estimate_h265_size_bits(model_modules, bit_depths=(8, 10, 12)):
    """
    Estimate H.265 compressed size for the non-quantize (center+normalize)
    export pipeline at multiple bit depths.

    For each DCT layer, simulates per-tile center+normalize to N-bit,
    then estimates compressed size via Shannon entropy of the resulting
    pixel distribution.  This matches the actual export_h265.py pipeline
    where each tile is independently normalized and losslessly encoded.

    Args:
        model_modules: iterable of (name, module) pairs from model.named_modules()
        bit_depths: tuple of bit depths to estimate

    Returns:
        dict mapping bit_depth -> estimated total bits
    """
    from dct_layers import DCTConv2d

    results = {b: 0.0 for b in bit_depths}
    raw_bytes = 0

    for name, m in model_modules:
        if not isinstance(m, DCTConv2d):
            continue

        w = m.weight_dct.data
        out_ch, in_ch, K_h, K_w = w.shape
        raw_bytes += w.numel() * 4  # float32

        # Reshape to 2D: (out_ch*K_h, in_ch*K_w)
        img = w.permute(0, 2, 1, 3).reshape(out_ch * K_h, in_ch * K_w)

        # Per-tile: simulate center+normalize and compute entropy
        # Use the same tile size logic as export_h265.py (max 128x128)
        h, w_dim = img.shape
        max_tile = 128
        min_dim = 8

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
                    tile = img[r:min(r+th, h), c:min(c+tw, w_dim)]
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
                    # All identical values -> near-zero entropy
                    results[b] += n_pixels * 0.1  # small overhead
                    continue

                norm_factor = max_val / span
                center = (fmin + fmax) / 2.0
                half = max_val / 2.0

                # Simulate normalization to integer pixels
                pixels = ((tile.float() - center) * norm_factor + half).round()
                pixels = pixels.clamp(0, max_val).long()

                # Compute Shannon entropy of the pixel distribution
                counts = torch.bincount(pixels.flatten(), minlength=max_val + 1)
                probs = counts.float() / n_pixels
                probs = probs[probs > 0]
                entropy = -(probs * probs.log2()).sum().item()

                # H.265 lossless typically achieves ~80-90% of Shannon entropy
                # due to CABAC context modeling overhead
                h265_efficiency = 1.15  # bits per pixel / entropy
                est_bits = n_pixels * entropy * h265_efficiency

                # Add per-tile metadata overhead (center + norm_factor)
                est_bits += 16 * 8  # 16 bytes for center + norm_factor

                results[b] += est_bits

    return results, raw_bytes
