from __future__ import annotations

import random
from typing import Any, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch_lightning.loggers.base import LoggerCollection
from pytorch_lightning.loggers.wandb import WandbLogger
from torchmetrics.functional import dice_score
from rich import print
from torch.optim.optimizer import Optimizer
from wandb.sdk.wandb_run import Run

from ..configs import Config
from ..models import UNet


def _dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2. * (pred * targs).sum() / (pred + targs).sum()


class Segmentor(pl.LightningModule):
    """
    Basic image classifier.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()  # type: ignore

        self.logger: Union[LoggerCollection, WandbLogger, Any]
        self.wandb: Run

        self.cfg = cfg

        self.model = UNet(self.cfg)
        self.criterion = nn.BCELoss()
        self.step = 0

    # -----------------------------------------------------------------------------------------------
    # Default PyTorch Lightning hooks
    # -----------------------------------------------------------------------------------------------
    def on_fit_start(self) -> None:
        """
        Hook before `trainer.fit()`.

        Attaches current wandb run to `self.wandb`.
        """
        if isinstance(self.logger, LoggerCollection):
            for logger in self.logger:  # type: ignore
                if isinstance(logger, WandbLogger):
                    self.wandb = logger.experiment  # type: ignore
        elif isinstance(self.logger, WandbLogger):
            self.wandb = self.logger.experiment  # type: ignore

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Hook on checkpoint saving.

        Adds config and RNG states to the checkpoint file.
        """
        checkpoint['cfg'] = self.cfg
        checkpoint['rng_torch'] = torch.default_generator.get_state()
        checkpoint['rng_numpy'] = np.random.get_state()
        checkpoint['rng_random'] = random.getstate()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Hook on checkpoint loading.

        Loads RNG states from the checkpoint file.
        """
        torch.default_generator.set_state(checkpoint['rng_torch'])
        np.random.set_state(checkpoint['rng_numpy'])
        random.setstate(checkpoint['rng_random'])

    # ----------------------------------------------------------------------------------------------
    # Optimizers
    # ----------------------------------------------------------------------------------------------
    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:  # type: ignore
        """
        Define system optimization procedure.

        See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers.

        Returns
        -------
        Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]
            Single optimizer or a combination of optimizers with learning rate schedulers.
        """
        optimizer: Optimizer = instantiate(
            self.cfg.optim.optimizer,
            params=self.parameters(),
            _convert_='all'
        )

        if self.cfg.optim.scheduler is not None:
            scheduler: _LRScheduler = instantiate(  # type: ignore
                self.cfg.optim.scheduler,
                optimizer=optimizer,
                _convert_='all'
            )
            print(optimizer, scheduler)
            return [optimizer], [scheduler]
        else:
            print(optimizer)
            return optimizer

    # ----------------------------------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Forward pass of the whole system.

        In this simple case just calls the main model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.model(x)

        # return torch.argmax(x, dim=1)
        return x

    # ----------------------------------------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------------------------------------
    def calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss value of a batch.

        In this simple case just forwards computation to default `self.criterion`.

        Parameters
        ----------
        outputs : torch.Tensor
            Network outputs with shape (batch_size, n_classes).
        targets : torch.Tensor
            Targets (ground-truth labels) with shape (batch_size).

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        if self.cfg.experiment.deep_supervision:
            return self.criterion(outputs[0], targets) + self.cfg.experiment.aux_loss_weight * self.criterion(outputs[1], targets)
        return self.criterion(outputs, targets)



    # ----------------------------------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------------------------------
    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:  # type: ignore
        """
        Train on a single batch with loss defined by `self.criterion`.

        Parameters
        ----------
        batch : list[torch.Tensor]
            Training batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        dict[str, torch.Tensor]
            Metric values for a given batch.
        """
        inputs, targets = batch
        outputs = self(inputs)  # basically equivalent to self.forward(data)
        loss = self.calculate_loss(outputs, targets)
        targets, outputs = targets.round().int(), outputs[0].round().int() if isinstance(outputs, tuple) else outputs.round().int()
        dice = _dice_score(outputs, targets)
        metrics = {
            'loss': loss,
            'dice_score': dice,
            # no need to return 'train_acc' here since it is always available as `self.train_acc`
        }
        self.step += 1
        if batch_idx % self.cfg.experiment.log_interval == 0:
            self.logger.log_metrics(metrics, step=float(self.step))
        return metrics

    def training_epoch_end(self, outputs: list[Any]) -> None:
        """
        Log training metrics.

        Parameters
        ----------
        outputs : list[Any]
            List of dictionaries returned by `self.training_step` with batch metrics.
        """
        step = self.current_epoch + 1

        metrics = {
            'epoch': float(step),
            # 'train_f1': float(self.train_f1.compute().item()),
        }

        # Average additional metrics over all batches
        for key in outputs[0]:
            metrics[key] = float(self._reduce(outputs, key).item())

        self.logger.log_metrics(metrics, step=step)

    def _reduce(self, outputs: list[Any], key: str):
        return torch.stack([out[key] for out in outputs]).mean().detach()

    # ----------------------------------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------------------------------
    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> dict[str, Any]:  # type: ignore
        """
        Compute validation metrics.

        Parameters
        ----------
        batch : list[torch.Tensor]
            Validation batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        dict[str, torch.Tensor]
            Metric values for a given batch.
        """

        inputs, targets = batch
        outputs = self(inputs)  # basically equivalent to self.forward(data)
        targets, outputs = targets.round().int(), outputs[0].round().int() if isinstance(outputs, tuple) else outputs.round().int()
        dice = dice_score(outputs, targets)

        return {
            'val_dice_score': dice,
            # 'additional_metric': ...
            # no need to return 'val_acc' here since it is always available as `self.val_acc`
        }

    def validation_epoch_end(self, outputs: list[Any]) -> None:
        """
        Log validation metrics.

        Parameters
        ----------
        outputs : list[Any]
            List of dictionaries returned by `self.validation_step` with batch metrics.
        """
        step = self.current_epoch + 1 if not self.trainer.running_sanity_check else self.current_epoch  # type: ignore

        metrics = {
            'epoch': float(step),
        }

        # Average additional metrics over all batches
        for key in outputs[0]:
            metrics[key] = float(self._reduce(outputs, key).item())
        self.logger.log_metrics(metrics, step=step)
