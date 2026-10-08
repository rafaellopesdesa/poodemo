"""Explicit sigmoid classifier with numerically stable ordinary BCE training.

The hidden MLP is the pinned toolkit architecture.  Its public output is a
probability; the loss and density-ratio conversion use the same pre-sigmoid
logit directly, avoiding precision loss from inverting a saturated sigmoid.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from nsbi_common_utils.lightning_tools import DensityRatioLightning


class LogLinearFloorLR(torch.optim.lr_scheduler.LRScheduler):
    """Decay geometrically by epoch, then train at an exact learning-rate floor.

    Epoch indices describe the rate *used* in that epoch. The last part of the
    configured training therefore really uses the floor, rather than merely
    reaching it in a scheduler update after the last optimizer step.
    """
    def __init__(self, optimizer, min_lr, total_epochs, decay_fraction=.9, last_epoch=-1):
        if total_epochs < 1:
            raise ValueError("total_epochs must be positive.")
        if not 0 < decay_fraction <= 1:
            raise ValueError("decay_fraction must be in (0, 1].")
        if not 0 < min_lr <= min(group["lr"] for group in optimizer.param_groups):
            raise ValueError("min_lr must be positive and no larger than the initial rate.")
        self.min_lr = float(min_lr)
        self.total_epochs = int(total_epochs)
        self.decay_fraction = float(decay_fraction)
        self.decay_epochs = int(math.ceil(decay_fraction * (total_epochs - 1)))
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.decay_epochs:
            return [self.min_lr for _ in self.base_lrs]
        fraction = self.last_epoch / self.decay_epochs
        return [math.exp((1 - fraction) * math.log(base) + fraction * math.log(self.min_lr))
                for base in self.base_lrs]


class SigmoidDensityRatioLightning(DensityRatioLightning):
    """Toolkit MLP + explicit sigmoid, without dropout or loss penalties."""
    def __init__(self, n_hidden=3, n_neurons=1024, input_dim=3,
                 learning_rate=1e-3, use_log_loss=True, activation="swish",
                 callback_factor=.5, callback_patience=15,
                 min_learning_rate=1e-9, lr_decay_fraction=.9, total_epochs=80):
        if not use_log_loss:
            raise ValueError("The sigmoid ratio classifier always trains with stable BCE logits.")
        super().__init__(n_hidden=n_hidden, n_neurons=n_neurons, input_dim=input_dim,
                         learning_rate=learning_rate, use_log_loss=True, activation=activation,
                         callback_factor=callback_factor, callback_patience=callback_patience)
        self.save_hyperparameters()
        self.sigmoid = nn.Sigmoid()

    def forward_logits(self, x):
        """Pre-sigmoid logit, also the raw log density ratio for equal priors."""
        return self.out(self.mlp(x))

    def forward(self, x):
        """Classification probability in [0, 1]."""
        return self.sigmoid(self.forward_logits(x))

    def _bce(self, batch):
        x, y, w = batch
        logits = self.forward_logits(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float().view(-1, 1), reduction="none")
        weights = w.float().view(-1, 1)
        return (loss * weights).sum() / weights.sum()

    def training_step(self, batch, batch_idx):
        loss = self._bce(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._bce(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr, weight_decay=0.)
        scheduler = LogLinearFloorLR(optimizer, self.hparams.min_learning_rate,
                                    self.hparams.total_epochs, self.hparams.lr_decay_fraction)
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": scheduler, "interval": "epoch", "frequency": 1}}
