"""Evaluator for LARGO: every sample is scored with one modality subset.

The subset is either fixed for the whole run (``mod=k``) or read per sample from the ``mod``
field of the datalist (``mod=None``), which is how the paper evaluates: each validation patient is
assigned one of the 15 subsets, uniformly distributed over the dataset.
"""

import torch
from ignite.engine import Engine
from monai.engines import SupervisedEvaluator

from .network import modality_combinations


def _unwrap(network):
    """Return the underlying model of a DistributedDataParallel / DataParallel wrapper."""
    return getattr(network, "module", network)


class LARGOEvaluator(SupervisedEvaluator):
    def __init__(self, num_modalities=4, mod=None, **kwargs):
        super().__init__(**kwargs)
        self.num_modalities = num_modalities
        self.combinations = modality_combinations(num_modalities)
        self.mod = mod

    def _mod_from_batch(self, batchdata):
        if self.mod is not None:
            return int(self.mod)
        if not (isinstance(batchdata, dict) and "mod" in batchdata):
            raise ValueError("No fixed `mod` given and the batch has no `mod` field; add `mod` to the datalist "
                             "or run with `--mod <index>`.")
        mod = batchdata["mod"]
        if isinstance(mod, torch.Tensor):
            if mod.numel() > 1 and len(mod.unique()) > 1:
                raise ValueError("All samples of a batch must share the same `mod`; use batch_size=1.")
            mod = mod.flatten()[0].item()
        return int(mod)

    def _iteration(self, engine: Engine, batchdata):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        mod = self._mod_from_batch(batchdata)
        batchdata = dict(batchdata)
        batchdata["image"] = batchdata["image"][:, self.combinations[mod]]
        _unwrap(engine.network).set_mod(mod)
        return super()._iteration(engine, batchdata)
