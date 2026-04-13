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


def calculate_group_lasso_penalty(
    weight_dct: torch.Tensor, threshold_freq: int = 1
) -> torch.Tensor:
    """
    L2,1 norm (Group Lasso) penalty on high-frequency DCT coefficients.

    Args:
        weight_dct: shape (out_channels, in_channels, K_h, K_w)
        threshold_freq: frequency index where "high frequency" starts.
            For a 3x3 kernel with threshold_freq=1, everything except
            the top-left 1x1 DC component is penalized.

    Returns:
        Scalar penalty tensor.
    """
    K_h = weight_dct.shape[-2]
    K_w = weight_dct.shape[-1]
    penalty = torch.tensor(0.0, device=weight_dct.device, dtype=weight_dct.dtype)

    for i in range(K_h):
        for j in range(K_w):
            if i >= threshold_freq or j >= threshold_freq:
                group = weight_dct[:, :, i, j]
                penalty = penalty + torch.norm(group, p=2)

    return penalty
