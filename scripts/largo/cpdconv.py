"""Low-rank convolutional layers (LRConv) based on the canonical polyadic (CP) decomposition.

Each layer represents the kernels of ``num_models`` dedicated networks jointly. The joint
weight tensor W of shape (M, C_in, C_out, K) is never materialised; instead its CP factors

    A (M x R)     model factor
    B (C_in x R)  input-channel factor
    C (C_out x R) output-channel factor
    D (K x R)     kernel factor

are trained directly from scratch. At run time the row ``A[m, :]`` of the available modality
subset ``m`` selects one model, and only that model's kernel is reconstructed:

    W^(m) = sum_r A[m, r] * B[:, r] o C[:, r] o D[:, r]

The rank R is chosen so that the layer has (approximately) the same number of parameters as a
single ordinary convolution, i.e. a compression ratio of 1/M with respect to M dedicated layers
(Eq. 3 of the paper).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def _triple(x):
    return (x, x, x) if isinstance(x, int) else tuple(x)


def cp_rank(in_channels, out_channels, kernel_numel, num_models):
    """Rank keeping the parameter count of one convolution: R = floor(C_in C_out K / (M + C_in + C_out + K))."""
    rank = math.floor(in_channels * out_channels * kernel_numel / (in_channels + out_channels + kernel_numel + num_models))
    return max(rank, 1)


class _CPDConvBase(nn.Module):
    """Shared parameterisation of the CP-factorised (transposed) convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, num_models=1, rank=None, bias=True,
                 checkpoint_kernels=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.num_models = num_models
        k3 = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.rank = cp_rank(in_channels, out_channels, k3, num_models) if rank is None else rank
        # Recompute the kernel reconstruction in the backward pass instead of storing its
        # (C_in x K x R) intermediate; saves memory at a negligible compute cost.
        self.checkpoint_kernels = checkpoint_kernels

        # CP factor matrices (names follow the paper: A = model, B = input, C = output, D = kernel)
        self.weight_model = nn.Parameter(torch.empty(num_models, self.rank))
        self.weight_input = nn.Parameter(torch.empty(in_channels, self.rank))
        self.weight_output = nn.Parameter(torch.empty(out_channels, self.rank))
        self.weight_kernel = nn.Parameter(torch.empty(k3, self.rank))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_models, out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # index of the modality subset whose kernel is reconstructed in ``forward``
        self.mod = 0

    def reset_parameters(self):
        for param in (self.weight_input, self.weight_output, self.weight_kernel):
            nn.init.kaiming_normal_(param, a=math.sqrt(5))
        # The B, C, D columns are normalised in ``forward``; the model factor carries the scale.
        with torch.no_grad():
            scale = math.sqrt(2 * self.out_channels / self.rank)
            self.weight_model.copy_(scale * (1 + 0.1 * torch.randn_like(self.weight_model)))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def _contract(a, b, c, d):
        """W[i, k, o] = sum_r a[r] b[i, r] c[o, r] d[k, r], contracting the small modes first."""
        eps = 1e-8
        b = b / (b.norm(dim=0, keepdim=True) + eps)
        c = c / (c.norm(dim=0, keepdim=True) + eps)
        d = d / (d.norm(dim=0, keepdim=True) + eps)
        ikr = b[:, None, :] * (d * a)[None, :, :]                 # (C_in, K, R)
        return ikr.reshape(-1, ikr.shape[-1]) @ c.t()             # (C_in * K, C_out)

    def reconstruct_kernel(self, mod, transposed=False):
        """Kernel of subset ``mod``: (C_out, C_in, *k) for a convolution, (C_in, C_out, *k) if ``transposed``."""
        factors = (self.weight_model[mod], self.weight_input, self.weight_output, self.weight_kernel)
        if self.checkpoint_kernels and torch.is_grad_enabled():
            w = torch.utils.checkpoint.checkpoint(self._contract, *factors, use_reentrant=False)
        else:
            w = self._contract(*factors)
        w = w.view(self.in_channels, -1, self.out_channels)       # (C_in, K, C_out)
        w = w.permute(0, 2, 1) if transposed else w.permute(2, 0, 1)
        return w.reshape(*w.shape[:2], *self.kernel_size)

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, num_models={self.num_models}, rank={self.rank}, "
            f"bias={self.bias is not None}"
        )


class CPDConv3d(_CPDConvBase):
    """Drop-in replacement of ``nn.Conv3d`` representing ``num_models`` kernels with CP factors."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, num_models=1, rank=None,
                 checkpoint_kernels=False):
        super().__init__(in_channels, out_channels, kernel_size, num_models=num_models, rank=rank, bias=bias,
                         checkpoint_kernels=checkpoint_kernels)
        self.stride = _triple(stride)
        self.padding = _triple(padding)

    def forward(self, x):
        mod = self.mod
        weight = self.reconstruct_kernel(mod)
        bias = self.bias[mod] if self.bias is not None else None
        return F.conv3d(x, weight, bias=bias, stride=self.stride, padding=self.padding)

    def extra_repr(self):
        return super().extra_repr() + f", stride={self.stride}, padding={self.padding}"


class CPDConvTranspose3d(_CPDConvBase):
    """Drop-in replacement of ``nn.ConvTranspose3d`` representing ``num_models`` kernels with CP factors."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True,
                 num_models=1, rank=None, checkpoint_kernels=False):
        super().__init__(in_channels, out_channels, kernel_size, num_models=num_models, rank=rank, bias=bias,
                         checkpoint_kernels=checkpoint_kernels)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.output_padding = _triple(output_padding)

    def forward(self, x):
        mod = self.mod
        weight = self.reconstruct_kernel(mod, transposed=True)
        bias = self.bias[mod] if self.bias is not None else None
        return F.conv_transpose3d(x, weight, bias=bias, stride=self.stride, padding=self.padding,
                                  output_padding=self.output_padding)

    def extra_repr(self):
        return super().extra_repr() + f", stride={self.stride}, padding={self.padding}, output_padding={self.output_padding}"
