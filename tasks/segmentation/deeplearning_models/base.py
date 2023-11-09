from typing import Tuple, Dict
from collections import defaultdict
from pytorch_lightning.utilities.seed import seed_everything
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics.functional import dice
import numpy as np
import wandb
import math 


from externel_library.mir_eval_simple.mir_eval_simple_utils import boundaries_to_intervals
from externel_library.mir_eval_simple.mir_eval_segment import pairwise, nce
import torch.optim as optim

class BaseModel(pl.LightningModule):
    """
    Base model for different Deep learning architecture
    train/validation/test step are implemented, as well as _test (used to calculate metrics)
    """
    def _predict(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor]) -> Tuple[torch.tensor, torch.tensor]:
      raise NotImplementedError()
   
    def evaluation_forward(self, x):
      raise NotImplementedError()
   
    def _test(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor]) -> Tuple[torch.tensor, Dict[str, float]]:
        """
        Perform the prediction step in the specified batch. When computing the loss
        masked elements are ignored.

        Args:
            batch (Tuple[torch.tensor, torch.tensor, torch.tensor]): 
            Input batch in the form (data, labels, padding mask)

        Returns:
            Tuple[torch.tensor, Dict[str, float]]: The loss item and the dictionary of metrics
        """
        metrics = defaultdict(list)
        mask = batch[-1]
        y = batch[-2]
        
        with torch.no_grad():
            pred, loss = self._predict(batch)
            
            for pi, yi, mi in zip(pred, y, mask):               
                pi = pi[mi != 0].argmax(axis=-1).cpu().numpy()
                _, pi = np.unique(pi, return_inverse=True)
            
                yi = yi[mi != 0].argmax(axis=-1).cpu().numpy()
                _, yi = np.unique(yi, return_inverse=True)

                # Calculate the time interval, in our task there're len(yi) different interval
                # e.g. when len(yi) = 2: intervals = [[0,1], [1,2]]
                # Obviously in our task, pi and yi share the same interval
                intervals = boundaries_to_intervals(np.arange(len(yi) + 1)) 
                
                # Calculate Precision, Recall Rate, F1-Score by using Agreement matrix
                precision, recall, f1 = pairwise(intervals, yi, intervals, pi)
                metrics["p_precision"].append(precision)
                metrics["p_recall"].append(recall)
                metrics["p_f1"].append(f1)

                # Caslculate S_u (under-segmentation score), S_o (over-segmentation score), and F1 for them 
                over, under, under_over_f1 = nce(intervals, yi, intervals, pi)
                metrics["under"] = under
                metrics["over"] = over
                metrics["under_over_f1"] = under_over_f1
        
        metrics = {k: np.mean(v) for k, v in metrics.items()}
        return loss, metrics


    def training_step(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor], batch_idx: int) -> torch.tensor:
        """
        Perform a training step.

        Args:
            batch (Tuple[torch.tensor, torch.tensor, torch.tensor]): Input batch composed of (data, label, padding mask).
            batch_idx (int): Batch index.

        Returns:
            torch.tensor: The torch item loss.
        """
        _, loss = self._predict(batch)
        self.log("train/loss", loss.item())
        wandb.log({"train/loss": loss})
        return loss


    def validation_step(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor], batch_idx: int) -> torch.tensor:
        """
        Perform a validation step.

        Args:
            batch (Tuple[torch.tensor, torch.tensor, torch.tensor]): Input batch composed of (data, label, padding mask).
            batch_idx (int): Batch index.

        Returns:
            torch.tensor: The torch item loss.
        """
        loss, metrics = self._test(batch)
        for k, m in metrics.items(): self.log(f"val_{k}", m)
        self.log("val/loss", loss.item())
        wandb.log({"val/loss": loss})
        return loss
    

    def test_step(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor], batch_idx: int) -> torch.tensor:
        """
        Perform a test step.

        Args:
            batch (Tuple[torch.tensor, torch.tensor, torch.tensor]): Input batch composed of (data, label, padding mask).
            batch_idx (int): Batch index.

        Returns:
            torch.tensor: The torch item loss.
        """
        loss, metrics = self._test(batch)        
        for k, m in metrics.items(): self.log(f"test_{k}", m)
        self.log("test/loss", loss.item())
        wandb.log({"test/loss": loss})
        return loss
