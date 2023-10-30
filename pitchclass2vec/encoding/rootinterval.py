from typing import List, Tuple
from itertools import chain
from functools import cache

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pitchclass2vec.data import ChocoDocumentDataset, ChocoChordDataset
from pitchclass2vec.encoding.utils import pitchclass_to_onehot, chord_pitchclass
from harte.harte import Harte

class RootIntervalMixin(object):
  @staticmethod
  @cache
  def encode_chord(chord: str) -> List[int]:
    """
    Encode a chord as the one-hot encoding of intervals between
    the root note and all the other notes in the chord.
    Empty chords are represented as all 0s.

    Args:
        chord (str): 

    Returns:
        List[np.array]: List of encoded components of the chord.
    """
    pc = chord_pitchclass(chord)

    if len(pc) != 0:
      intervals = [(pc[0], elem) for elem in pc]
      onehot = list(map(pitchclass_to_onehot, intervals))
    else:
      onehot = [pitchclass_to_onehot(pc)]

    encoding = list(map(lambda x: x.dot(2**np.arange(12)[::-1]).astype(int), onehot))
    return encoding


class TimedRootIntervalDataset(RootIntervalMixin, ChocoDocumentDataset):
  @staticmethod
  def collate_fn(sample: Tuple[List[np.array], List[np.array], np.array]) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Collate the provided samples in a single batch.

    Args:
        sample (List[List[np.array], List[np.array], np.array]): Input sample

    Returns:
        Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array], np.array]: Output batch.
    """
    source, target, y, duration = zip(*sample)
    
    source = torch.nn.utils.rnn.pad_sequence(map(torch.tensor, chain(*source)), 
                                             batch_first=True, 
                                             padding_value=0).int()
    
    target = torch.nn.utils.rnn.pad_sequence(map(torch.tensor, chain(*target)), 
                                             batch_first=True, 
                                             padding_value=0).int()
   
    duration = torch.tensor(list(chain(*duration)))
    y = torch.tensor(list(chain(*y)))

    return source, target, y, duration


class RootIntervalDataset(RootIntervalMixin, ChocoChordDataset):
  @staticmethod
  def collate_fn(sample: Tuple[List[np.array], List[np.array], np.array]) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Collate the provided samples in a single batch.

    Args:
        sample (List[List[np.array], List[np.array], np.array]): Input sample

    Returns:
        Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array], np.array]: Output batch.
    """
    source, target, y = zip(*sample)
    
    source = torch.nn.utils.rnn.pad_sequence(map(torch.tensor, chain(*source)), 
                                             batch_first=True, 
                                             padding_value=0).int()
    
    target = torch.nn.utils.rnn.pad_sequence(map(torch.tensor, chain(*target)), 
                                             batch_first=True, 
                                             padding_value=0).int()
    
    y = torch.tensor(list(chain(*y)))

    return source, target, y
