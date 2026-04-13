import torch
import torch.nn as nn
import torch.nn.functional as F
from dct_utils import get_1d_dct_matrix


class DCTConv2d(nn.Module):
    """
    Convolutional layer whose learnable parameters live in the DCT (frequency)
    domain.  The forward pass materializes spatial weights via 2D IDCT before
    running a standard conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        K_h, K_w = kernel_size

        # Learnable DCT coefficients (W_hat)
        self.weight_dct = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, K_h, K_w)
        )
        nn.init.kaiming_uniform_(self.weight_dct, a=5**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # Precompute IDCT matrices: spatial_W = C_h^T @ W_hat @ C_w
        C_h = get_1d_dct_matrix(K_h)
        C_w = get_1d_dct_matrix(K_w)
        self.register_buffer("C_h_T", C_h.t())
        self.register_buffer("C_w", C_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Materialize spatial weights via 2D IDCT
        spatial_weight = torch.einsum(
            "ab, oibd, cd -> oiac", self.C_h_T, self.weight_dct, self.C_w
        )

        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                spatial_weight,
                self.bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )

        return F.conv2d(
            x, spatial_weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}"
        )


def replace_with_dct_convs(module: nn.Module) -> None:
    """
    Recursively replace all nn.Conv2d layers (kernel > 1x1) with DCTConv2d.
    1x1 pointwise convolutions are left unchanged — they have no spatial
    frequencies to compress.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if child.kernel_size == (1, 1):
                continue

            dct_conv = DCTConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
                padding_mode=child.padding_mode,
            )
            setattr(module, name, dct_conv)
        else:
            replace_with_dct_convs(child)
