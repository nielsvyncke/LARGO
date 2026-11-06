from typing import Optional, List, Dict, Any
from copy import deepcopy
import logging
import os

import torch
from ignite.handlers import Checkpoint
from ignite.engine import Events, Engine
from monai.handlers import CheckpointLoader as _MonaiCheckpointLoader

class CheckpointLoader(_MonaiCheckpointLoader):
    """
    Custom CheckpointLoader that handles PyTorch 2.6+ weights_only parameter.
    Wraps MONAI's CheckpointLoader to add weights_only=False for compatibility.
    """
    
    def __call__(self, engine: Engine) -> None:
        """Load checkpoint with weights_only=False for PyTorch 2.6+ compatibility."""
        if not os.path.exists(self.load_path):
            return
        
        # Load checkpoint with weights_only=False
        checkpoint = torch.load(
            self.load_path, 
            map_location=self.map_location,
            weights_only=False  # Required for PyTorch 2.6+
        )
        
        # Load objects from checkpoint
        Checkpoint.load_objects(to_load=self.load_dict, checkpoint=checkpoint)
        
        if self.logger is not None:
            self.logger.info(f"Loaded checkpoint from {self.load_path}")


def load_checkpoint(
    state_dict: Dict[str, Any],
    path: str,
    state_dict_options: Optional[Dict[str, Any]] = None,
    checkpoint_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    state_dict_options = state_dict_options or {}
    checkpoint_options = checkpoint_options or {}

    new_state_dict = deepcopy(state_dict)

    checkpoint_data = torch.load(path, weights_only=False, **checkpoint_options)
    Checkpoint.load_objects(new_state_dict, checkpoint_data, **state_dict_options)

    return new_state_dict


def load_checkpoints(
    state_dict: Dict[str, Any], path_list: List[str], **kwargs: Dict[str, Any]
) -> List[Dict[str, Any]]:
    return [load_checkpoint(state_dict, path, **kwargs) for path in path_list]


class ValidationEarlyStopHandler:
    def __init__(self, patience=50, min_delta=0.001, metric_name="val_mean_dice", evaluator=None):
        """
        Early stopping handler that works with MONAI validation intervals.
        
        Args:
            patience: Number of validation cycles to wait for improvement
            min_delta: Minimum change to qualify as improvement
            metric_name: Name of the metric to monitor
            evaluator: The evaluator instance to attach to
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.evaluator = evaluator
        self.best_score = None
        self.wait_count = 0
        self.trainer = None
        self.logger = logging.getLogger(__name__)
        self.attached = False

    def attach(self, trainer):
        """Attach to trainer."""
        self.trainer = trainer
        
        # If we have direct access to evaluator, attach to it
        if self.evaluator:
            self.evaluator.add_event_handler(Events.COMPLETED, self.check_validation_metrics)

    def check_validation_metrics(self, evaluator):
        """Check validation metrics if available."""            
        # Check if we have validation metrics in evaluator state
        if hasattr(evaluator.state, 'metrics') and self.metric_name in evaluator.state.metrics:
            current_score = evaluator.state.metrics[self.metric_name]
            
            if self.best_score is None:
                self.best_score = current_score
                self.wait_count = 0
                self.logger.info(f"Initial {self.metric_name}: {current_score:.4f}")
            elif current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.wait_count = 0
                self.logger.info(f"New best {self.metric_name}: {current_score:.4f}")
            else:
                self.wait_count += 1
                self.logger.info(f"No improvement in {self.metric_name}. Wait count: {self.wait_count}/{self.patience}")
                
            if self.wait_count >= self.patience:
                self.logger.info(f"Early stopping triggered after {self.wait_count} validation cycles without improvement")
                if self.trainer:
                    self.trainer.terminate()
        else:
            print(f"Metric '{self.metric_name}' not found in evaluator.state.metrics")

class LogModelInfoHandler:
    def __init__(self, model):
        """
        Initializes the LogModelInfoHandler with the given model.

        Args:
            model (torch.nn.Module): The model to log information about.
        """
        self.model = model
        self.logger = logging.getLogger(__name__)

    def attach(self, engine):
        """
        Attaches the handler to the Ignite engine.

        Args:
            engine (ignite.engine.Engine): The Ignite engine to attach the handler to.
        """
        engine.add_event_handler(Events.STARTED, self.log_model_info)

    def log_model_info(self, engine):
        """
        Logs model information, such as total parameters, trainable parameters, and non-trainable parameters.

        Args:
            engine (ignite.engine.Engine): The Ignite engine.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        non_trainable_params = total_params - trainable_params

        self.logger.info("==========================")
        self.logger.info(f"Model: {self.model._get_name()}")
        self.logger.info("==========================")
        self.logger.info(f"Total params (M): {total_params / 1e6:.1f}")
        self.logger.info(f"Trainable params (M): {trainable_params / 1e6:.1f}")
        self.logger.info(f"Non-trainable params: {non_trainable_params}")
        self.logger.info("==========================")
