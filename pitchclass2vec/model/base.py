from typing import List, Tuple, Dict
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
  def _predict(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Perform a prediction step by encoding both the source and the target
    using the embedding method and compute the dot product between the
    vectors.

    Args:
        source (torch.Tensor): Source chords
        target (torch.Tensor): Target chords

    Returns:
        torch.Tensor: Tensor of similarities between chord computes as dot products
    """
    raise NotImplementedError()

  def training_step(self, batch: torch.Tensor, batch_idx: int):
    """
    Perform a training step on the provided batch.

    Args:
        batch (torch.Tensor): The provided batch.
        batch_idx (int): The index of the provided batch.

    Returns:
        Loss on the provided batch.
    """
    raise NotImplementedError()

  def configure_optimizers(self) -> Tuple[List[torch.optim.Adam], Dict]:
    """
    Lightning optimizers configuration. Uses Adam with 0.1 initial
    learning rate and a scheduler that reduces the learning rate
    after 10 optimizations without any improvement in the training
    loss.

    Returns:
        Tuple[List[torch.optim.Adam], Dict]: The optimizer and the relative scheduler.
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
    scheduler = {
      "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2),
      "monitor": "train/loss",
      "interval": "epoch" }
    return [optimizer], scheduler

  def __getitem__(self, chord: np.array) -> np.array:
    """
    Compute the embedding of a chord.

    Args:
        chord (np.array): Input chord already encoded

    Returns:
        np.array: Embedded chord.
    """
    with torch.no_grad():
      embedded = self.embedding(torch.tensor(chord).unsqueeze(0)).squeeze(0).numpy()

    return embedded