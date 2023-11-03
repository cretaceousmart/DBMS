from typing import Tuple, Dict
from collections import defaultdict
from pytorch_lightning.utilities.seed import seed_everything
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics.functional import dice
import numpy as np
import wandb

from mir_eval.util import boundaries_to_intervals
from mir_eval.segment import pairwise, nce

class DiceLoss(nn.Module):
  """
  Implement the Dice loss
  """
  def forward(self, inputs, targets, smooth=1):
      #flatten label and prediction tensors
      inputs = inputs.view(-1)
      targets = targets.view(-1)

      intersection = (inputs * targets).sum()                            
      dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

      return 1 - dice

class LSTMBaselineModel(pl.LightningModule):
  def __init__(self, 
               segmentation_train_args: dict = None,
               embedding_dim: int = 10,
               hidden_size: int = 100, 
               dropout: float = 0.0,
               num_layers: int = 1, 
               num_labels: int = 11,
               learning_rate: float = 0.001
               ):
    """
    Model for the functional segmentation of a music piece.

    Args:
        embedding_dim (int, optional): Dimension of the chord embeddings. Defaults to 10.
        hidden_size (int, optional): LSTM hidden size. Defaults to 100.
        dropout (float, optional): LSTM dropout. Defaults to 0.0.
        num_layers (int, optional): Number of stacked layers in the LSTM. Defaults to 1.
        num_labels (int, optional): Number of sections to be predicted. Defaults to 11.
        learning_rate (float, optional): Default learning rate. Defaults to 0.001.
    """
    super().__init__()
    self.save_hyperparameters()
    self.learning_rate = learning_rate
    self.lstm = nn.LSTM(embedding_dim,
                        hidden_size,
                        dropout=dropout,
                        num_layers=num_layers,
                        bidirectional=True,
                        batch_first=True)
    self.classification = nn.Linear(hidden_size * 2, num_labels)
    self.softmax = nn.Softmax(dim=2)
    self.segmentation_train_args = segmentation_train_args
      
  def _predict(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor]) -> Tuple[torch.tensor, torch.tensor]:
    """
    Perform the prediction step in the specified batch. When computing the loss
    masked elements are ignored.

    Args:
        batch (Tuple[torch.tensor, torch.tensor, torch.tensor]): 
          Input batch in the form (data, labels, padding mask)

    Returns:
        Tuple[torch.tensor, torch.tensor]: The prediction and the loss item
    """
    x, y, mask = batch
    x, _ = self.lstm(x)
    x = self.classification(x)
    x = self.softmax(x)

    try:
        x, y, mask = batch
        x, _ = self.lstm(x)
        x = self.classification(x)
        x = self.softmax(x)

        # print("Jie Log：x:", x.shape)
        # print("Jie Log：y:", y.shape)
        # print("Jie Log：mask:", mask.shape)
        # print("Jie Log：x[mask != 0].float() shape:", x[mask != 0].float().shape)
        # print("Jie Log：y[mask != 0].float() shape:", y[mask != 0].float().shape)

        loss = nn.functional.binary_cross_entropy(x[mask != 0].float(), y[mask != 0].float())
    
    except Exception as e:
        print("An error occurred:", e)
        raise e  # re-raise the error after printing it out

    return x, loss

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
        
            intervals = boundaries_to_intervals(np.arange(len(yi) + 1))
            precision, recall, f1 = pairwise(intervals, yi, intervals, pi)
            metrics["p_precision"].append(precision)
            metrics["p_recall"].append(recall)
            metrics["p_f1"].append(f1)
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

  def configure_optimizers(self) -> Dict:
    """
    Configure the optimizers such that after 5 epochs without improvement the 
    learning rate is decreased.

    Returns:
        Dict: Optimizer configuration
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=self.segmentation_train_args.get("factor",0.1),
            patience=self.segmentation_train_args.get("patience",5)
            )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val/loss"
        }
    }

  #----- Add some function -----------------------------------------------------------------

import collections
import warnings

import numpy as np
import scipy.stats
import scipy.sparse
import scipy.misc
import scipy.special
import sys
sys.path.append("/app/mir_eval_simple/")
import mir_eval_simple_utils





def validate_structure(reference_intervals, reference_labels,
                       estimated_intervals, estimated_labels):
    """Checks that the input annotations to a structure estimation metric (i.e.
    one that takes in both segment boundaries and their labels) look like valid
    segment times and labels, and throws helpful errors if not.

    Parameters
    ----------
    reference_intervals : np.ndarray, shape=(n, 2)
        reference segment intervals, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.

    reference_labels : list, shape=(n,)
        reference segment labels, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.

    estimated_intervals : np.ndarray, shape=(m, 2)
        estimated segment intervals, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.

    estimated_labels : list, shape=(m,)
        estimated segment labels, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.

    """
    for (intervals, labels) in [(reference_intervals, reference_labels),
                                (estimated_intervals, estimated_labels)]:

        mir_eval_simple_utils.validate_intervals(intervals)
        if intervals.shape[0] != len(labels):
            raise ValueError('Number of intervals does not match number '
                             'of labels')

        # Check only when intervals are non-empty
        if intervals.size > 0:
            # Make sure intervals start at 0
            if not np.allclose(intervals.min(), 0.0):
                raise ValueError('Segment intervals do not start at 0')

    if reference_intervals.size == 0:
        warnings.warn("Reference intervals are empty.")
    if estimated_intervals.size == 0:
        warnings.warn("Estimated intervals are empty.")
    # Check only when intervals are non-empty
    if reference_intervals.size > 0 and estimated_intervals.size > 0:
        if not np.allclose(reference_intervals.max(),
                           estimated_intervals.max()):
            raise ValueError('End times do not match')

def _contingency_matrix(reference_indices, estimated_indices):
    """Computes the contingency matrix of a true labeling vs an estimated one.

    Parameters
    ----------
    reference_indices : np.ndarray
        Array of reference indices
    estimated_indices : np.ndarray
        Array of estimated indices

    Returns
    -------
    contingency_matrix : np.ndarray
        Contingency matrix, shape=(#reference indices, #estimated indices)
    .. note:: Based on sklearn.metrics.cluster.contingency_matrix

    """
    ref_classes, ref_class_idx = np.unique(reference_indices,
                                           return_inverse=True)
    est_classes, est_class_idx = np.unique(estimated_indices,
                                           return_inverse=True)
    n_ref_classes = ref_classes.shape[0]
    n_est_classes = est_classes.shape[0]
    # Using coo_matrix is faster than histogram2d
    return scipy.sparse.coo_matrix((np.ones(ref_class_idx.shape[0]),
                                    (ref_class_idx, est_class_idx)),
                                   shape=(n_ref_classes, n_est_classes),
                                   dtype=np.int64).toarray()


def nce(reference_intervals, reference_labels, estimated_intervals,
      estimated_labels, frame_size=0.1, beta=1.0, marginal=False):
  """Frame-clustering segmentation: normalized conditional entropy

  Computes cross-entropy of cluster assignment, normalized by the
  max-entropy.

  Examples
  --------
  >>> (ref_intervals,
  ...  ref_labels) = mir_eval.io.load_labeled_intervals('ref.lab')
  >>> (est_intervals,
  ...  est_labels) = mir_eval.io.load_labeled_intervals('est.lab')
  >>> # Trim or pad the estimate to match reference timing
  >>> (ref_intervals,
  ...  ref_labels) = mir_eval.util.adjust_intervals(ref_intervals,
  ...                                               ref_labels,
  ...                                               t_min=0)
  >>> (est_intervals,
  ...  est_labels) = mir_eval.util.adjust_intervals(
  ...     est_intervals, est_labels, t_min=0, t_max=ref_intervals.max())
  >>> S_over, S_under, S_F = mir_eval.structure.nce(ref_intervals,
  ...                                               ref_labels,
  ...                                               est_intervals,
  ...                                               est_labels)

  Parameters
  ----------
  reference_intervals : np.ndarray, shape=(n, 2)
      reference segment intervals, in the format returned by
      :func:`mir_eval.io.load_labeled_intervals`.
  reference_labels : list, shape=(n,)
      reference segment labels, in the format returned by
      :func:`mir_eval.io.load_labeled_intervals`.
  estimated_intervals : np.ndarray, shape=(m, 2)
      estimated segment intervals, in the format returned by
      :func:`mir_eval.io.load_labeled_intervals`.
  estimated_labels : list, shape=(m,)
      estimated segment labels, in the format returned by
      :func:`mir_eval.io.load_labeled_intervals`.
  frame_size : float > 0
      length (in seconds) of frames for clustering
      (Default value = 0.1)
  beta : float > 0
      beta for F-measure
      (Default value = 1.0)

  marginal : bool
      If `False`, normalize conditional entropy by uniform entropy.
      If `True`, normalize conditional entropy by the marginal entropy.
      (Default value = False)

  Returns
  -------
  S_over
      Over-clustering score:

      - For `marginal=False`, ``1 - H(y_est | y_ref) / log(|y_est|)``

      - For `marginal=True`, ``1 - H(y_est | y_ref) / H(y_est)``

      If `|y_est|==1`, then `S_over` will be 0.

  S_under
      Under-clustering score:

      - For `marginal=False`, ``1 - H(y_ref | y_est) / log(|y_ref|)``

      - For `marginal=True`, ``1 - H(y_ref | y_est) / H(y_ref)``

      If `|y_ref|==1`, then `S_under` will be 0.

  S_F
      F-measure for (S_over, S_under)

  """

  validate_structure(reference_intervals, reference_labels,
                      estimated_intervals, estimated_labels)

  # Check for empty annotations.  Don't need to check labels because
  # validate_structure makes sure they're the same size as intervals
  if reference_intervals.size == 0 or estimated_intervals.size == 0:
      return 0., 0., 0.

  # Generate the cluster labels
  y_ref = mir_eval_simple_utils.intervals_to_samples(reference_intervals,
                                    reference_labels,
                                    sample_size=frame_size)[-1]

  y_ref = mir_eval_simple_utils.index_labels(y_ref)[0]

  # Map to index space
  y_est = mir_eval_simple_utils.intervals_to_samples(estimated_intervals,
                                    estimated_labels,
                                    sample_size=frame_size)[-1]

  y_est = mir_eval_simple_utils.index_labels(y_est)[0]

  # Make the contingency table: shape = (n_ref, n_est)
  contingency = _contingency_matrix(y_ref, y_est).astype(float)

  # Normalize by the number of frames
  contingency = contingency / len(y_ref)

  # Compute the marginals
  p_est = contingency.sum(axis=0)
  p_ref = contingency.sum(axis=1)

  # H(true | prediction) = sum_j P[estimated = j] *
  # sum_i P[true = i | estimated = j] log P[true = i | estimated = j]
  # entropy sums over axis=0, which is true labels

  true_given_est = p_est.dot(scipy.stats.entropy(contingency, base=2))
  pred_given_ref = p_ref.dot(scipy.stats.entropy(contingency.T, base=2))

  if marginal:
      # Normalize conditional entropy by marginal entropy
      z_ref = scipy.stats.entropy(p_ref, base=2)
      z_est = scipy.stats.entropy(p_est, base=2)
  else:
      z_ref = np.log2(contingency.shape[0])
      z_est = np.log2(contingency.shape[1])

  score_under = 0.0
  if z_ref > 0:
      score_under = 1. - true_given_est / z_ref

  score_over = 0.0
  if z_est > 0:
      score_over = 1. - pred_given_ref / z_est

  f_measure = mir_eval_simple_utils.f_measure(score_over, score_under, beta=beta)

  return score_over, score_under, f_measure
