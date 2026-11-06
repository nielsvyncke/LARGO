"""LARGO: a low-rank hypernetwork that compresses the 2^N - 1 dedicated missing-modality
networks of an nnU-Net backbone into a single model.

* ``Stem``: one small uncompressed convolution per modality subset (the number of input
  channels differs per subset).
* Inner convolutional layers: every ``nn.Conv3d`` / ``nn.ConvTranspose3d`` of the MONAI
  ``DynUNet`` is replaced by its CP-factorised counterpart (``CPDConv3d`` /
  ``CPDConvTranspose3d``) that holds the kernels of all subsets.
* ``Head``: one uncompressed 1x1x1 output convolution per subset.

Selecting a subset is a plain index: ``network.mod = m`` before calling ``forward``.
"""

from copy import deepcopy
from itertools import combinations

import torch.nn as nn
from monai.networks.nets import DynUNet

from .cpdconv import CPDConv3d, CPDConvTranspose3d


def modality_combinations(num_modalities):
    """All non-empty subsets of ``range(num_modalities)``, ordered by size then lexicographically.

    For four modalities this gives the 15 subsets indexed by ``mod`` in the datalist and configs:
    0: [0], 1: [1], 2: [2], 3: [3], 4: [0, 1], ..., 14: [0, 1, 2, 3].
    """
    return [list(c) for k in range(1, num_modalities + 1) for c in combinations(range(num_modalities), k)]


class Stem(nn.Module):
    """Uncompressed first convolution, one per modality subset (input channel count varies)."""

    def __init__(self, num_modalities, out_channels, kernel_size=3):
        super().__init__()
        self.combinations = modality_combinations(num_modalities)
        self.layers = nn.ModuleList(
            nn.Conv3d(len(comb), out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            for comb in self.combinations
        )

    def forward(self, x, mod):
        return self.layers[mod](x)


class Head(nn.Module):
    """Uncompressed output block, one copy per modality subset."""

    def __init__(self, output_block, num_models):
        super().__init__()
        self.blocks = nn.ModuleList(deepcopy(output_block) for _ in range(num_models))

    def forward(self, x, mod):
        return self.blocks[mod](x)


class LARGO(DynUNet):
    """nnU-Net (MONAI ``DynUNet``) whose inner convolutions are CP-factorised over the 2^N - 1
    modality subsets.

    Args:
        rank: CP rank of every low-rank layer. ``None`` (default) picks the rank of Eq. 3 of the
            paper so that the whole hypernetwork has the parameter budget of one nnU-Net.
        checkpoint_kernels: recompute the kernel reconstruction during backward instead of storing
            its intermediate (lower training memory, slightly more compute).
        in_channels: number of modalities N. The input of ``forward`` must contain only the
            channels of the selected subset, i.e. ``len(combinations[mod])`` channels.
        Remaining arguments are forwarded to ``monai.networks.nets.DynUNet``.
    """

    def __init__(self, rank=None, checkpoint_kernels=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.combinations = modality_combinations(self.in_channels)
        self.num_models = len(self.combinations)
        self.rank = rank
        self.lowrank_layers = []
        self.mod = self.num_models - 1  # default: all modalities available

        def to_lowrank(module):
            if isinstance(module, nn.Conv3d):
                new = CPDConv3d(module.in_channels, module.out_channels, module.kernel_size, stride=module.stride,
                                padding=module.padding, bias=module.bias is not None, num_models=self.num_models,
                                rank=rank, checkpoint_kernels=checkpoint_kernels)
            elif isinstance(module, nn.ConvTranspose3d):
                new = CPDConvTranspose3d(module.in_channels, module.out_channels, module.kernel_size,
                                         stride=module.stride, padding=module.padding,
                                         output_padding=module.output_padding, bias=module.bias is not None,
                                         num_models=self.num_models, rank=rank, checkpoint_kernels=checkpoint_kernels)
            else:
                for name, child in module.named_children():
                    setattr(module, name, to_lowrank(child))
                return module
            self.lowrank_layers.append(new)
            return new

        # The first convolution of the input block becomes the per-subset stem.
        self.input_block.conv1 = nn.Identity()
        for block in (self.input_block, self.downsamples, self.bottleneck, self.upsamples):
            to_lowrank(block)
        self.stem = Stem(self.in_channels, self.filters[0], kernel_size=self.kernel_size[0][0]
                         if isinstance(self.kernel_size[0], (list, tuple)) else self.kernel_size[0])
        self.head = Head(self.output_block, self.num_models)
        self.output_block = nn.Identity()

    def set_mod(self, mod):
        """Select the modality subset (index into ``self.combinations``) for the next forward pass."""
        if not 0 <= mod < self.num_models:
            raise ValueError(f"mod must be in [0, {self.num_models - 1}], got {mod}")
        self.mod = mod
        for layer in self.lowrank_layers:
            layer.mod = mod

    def forward(self, x):
        mod = self.mod
        for layer in self.lowrank_layers:
            layer.mod = mod
        x = self.stem(x, mod)
        x = super().forward(x)
        return self.head(x, mod)
