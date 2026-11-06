"""Trainer for LARGO with random modality dropout and full-modality guidance (Eq. 5 of the paper).

Every iteration draws one random modality subset m ~ U(1, M) and computes

    L = 1/2 ( L_seg(H(X_m; m), Y) + L_seg(H(X; M), Y) ),

i.e. the segmentation loss of the random subset plus the loss of the full-modality input.
"""

import torch
from ignite.engine import Engine
from monai.engines import SupervisedTrainer
from monai.engines.utils import IterationEvents
from monai.utils import CommonKeys as Keys

from .network import modality_combinations


def _unwrap(network):
    """Return the underlying model of a DistributedDataParallel / DataParallel wrapper."""
    return getattr(network, "module", network)


class LARGOTrainer(SupervisedTrainer):
    """``SupervisedTrainer`` whose iteration applies modality dropout and selects the matching subset."""

    def __init__(self, num_modalities=4, max_grad_norm=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_modalities = num_modalities
        self.max_grad_norm = max_grad_norm
        self.combinations = modality_combinations(num_modalities)

    def _select(self, inputs, mod):
        """Keep only the channels of subset ``mod``."""
        return inputs[:, self.combinations[mod]]

    def _iteration(self, engine: Engine, batchdata):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 2:
            inputs, targets = batch
            args, kwargs = (), {}
        else:
            inputs, targets, args, kwargs = batch

        random_mod = int(torch.randint(len(self.combinations), (1,)).item())
        full_mod = len(self.combinations) - 1

        engine.network.train()
        engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

        def forward_and_loss(mod):
            _unwrap(engine.network).set_mod(mod)
            pred = engine.inferer(self._select(inputs, mod), engine.network, *args, **kwargs)
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            return pred, engine.loss_function(pred, targets).mean()

        if engine.amp and engine.scaler is not None:
            with torch.autocast("cuda", **engine.amp_kwargs):
                random_pred, random_loss = forward_and_loss(random_mod)
                _, full_loss = forward_and_loss(full_mod)
                loss = (random_loss + full_loss) / 2
        else:
            random_pred, random_loss = forward_and_loss(random_mod)
            _, full_loss = forward_and_loss(full_mod)
            loss = (random_loss + full_loss) / 2

        engine.state.output = {
            Keys.IMAGE: inputs,
            Keys.LABEL: targets,
            Keys.PRED: random_pred,
            Keys.LOSS: loss,
            "loss_random": random_loss.detach(),
            "loss_full": full_loss.detach(),
            "mod": random_mod,
        }
        engine.fire_event(IterationEvents.LOSS_COMPLETED)

        if engine.amp and engine.scaler is not None:
            engine.scaler.scale(loss).backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            engine.scaler.unscale_(engine.optimizer)
            torch.nn.utils.clip_grad_norm_(engine.network.parameters(), self.max_grad_norm)
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
        else:
            loss.backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            torch.nn.utils.clip_grad_norm_(engine.network.parameters(), self.max_grad_norm)
            engine.optimizer.step()

        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        return engine.state.output
