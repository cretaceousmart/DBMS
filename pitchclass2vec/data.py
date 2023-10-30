from typing import List, Tuple, Dict
import os
from collections import Counter
from itertools import chain, repeat
import pathlib
import pickle

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm

from choco.corpus import ChoCoDocument, ChoCoValidHarteChordsCorpus
import jams

CACHE_PATH = pathlib.Path(__file__).parent / ".cache"
if not os.path.exists(CACHE_PATH):
  os.makedirs(CACHE_PATH)

class ChocoDocumentDataset(torch.utils.data.Dataset):
  def __init__(self, corpus: List[ChoCoDocument], 
               context_size: int = 3,
               negative_sampling_k: int = 20):
    """
    Args:
        corpus (List[ChoCoDocument]): Documents from ChoCo
        context_size (int, optional): Context size for each word. The final window will result in 2*context_size+1. Defaults to 3.
        negative_sampling_k (int, optional): Number of negatively sampled examples. Defaults to 20.
    """
    super().__init__()

    self._cache_file = CACHE_PATH / "corpus.pickle"

    if os.path.exists(self._cache_file):
      
      with open(self._cache_file, "rb") as f:
        self.corpus = pickle.load(f)
    else:
      self.corpus = list(tqdm(corpus))

      with open(self._cache_file, "wb") as f:
        pickle.dump(self.corpus, f)
      
    self.c = context_size
    self.k = negative_sampling_k

    # Retrieval of a single sample means taking a specific chord of a specific document.
    # To efficiently index in this structure we keep a list in which every element is
    # represents the final index of each document if we flattened the whole corpus
    # in a single array. In this way we can easily search from which document a sample
    # comes from.
    self._doc_buckets = np.cumsum([0] + [len(doc) for doc in self.corpus])

    self.vocab = np.array(sorted(set((ann.symbol for doc in self.corpus for ann in doc.annotations))))


  def __len__(self) -> int:
    """
    Returns:
        int: Length of the dataset, defined as the number of chord occurences in the corpus.
    """
    return sum(len(doc) for doc in self.corpus)

  def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[int], List[float]]:
    """
    Retrieve an item from the dataset and perform negative sampling on it.
    Duration for negative samples is randomly sampled from a normal distribution
    whose meand and standard deviation are taken from the document duration
    distribution.

    Args:
        idx (int): The item index.

    Returns:
        Tuple[List[int], List[int], List[int], List[float]]: The current item, positive examples, negative examples and durations.
    """
    # retrieve the document index by finding the bucket in which the current index falls in
    bucket_idx = np.searchsorted(self._doc_buckets, idx, "right")
    doc = np.array(self.corpus[bucket_idx - 1].annotations)
    doc_len = len(doc)
    # retrieve the by computing the relative position to the found bucket
    elem_idx = idx - self._doc_buckets[bucket_idx - 1]

    # take the positive examples: i.e. those documents that falls within the context window
    if self.c == -1:
      # infinite context
      positive_idxs = np.arange(doc_len)
    else:
      left_context_elems = max(-1 * elem_idx, -1 * self.c) # available left elements
      right_context_elems = min(doc_len - elem_idx, self.c) # available right elements
      positive_idxs = np.arange(left_context_elems, right_context_elems) + elem_idx

    # negative examples are indexes that falls out of the positive indexes
    # based on https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html
    num_negatives = len(self.vocab) - (2 * self.c + 1) if self.c != -1 else len(self.vocab)
    negative_idxs = np.random.randint(0, num_negatives, size=self.k)
    if self.c != -1:
      pos_idxs_adj = positive_idxs - np.arange(len(positive_idxs))
      negative_idxs = negative_idxs + np.searchsorted(pos_idxs_adj, negative_idxs, side='right')
    
    # compute target as the set of encoded positives and negatives
    # and compute the label y as 1 for positives and 0 for negatives
    target = [ self.encode_chord(c) for c in doc[positive_idxs, 0] ]
    target += [ self.encode_chord(c) for c in self.vocab[negative_idxs] ]
    
    doc_durations = doc[:, 1].astype(float)
    duration = doc_durations[positive_idxs]
    duration = (duration - duration.min())/(duration.max() - duration.min()) \
               if duration.std() > 0 else np.ones_like(duration)
    duration = duration.tolist()
    duration += np.random.normal(doc_durations.mean(), doc_durations.std(), self.k).tolist()
    
    y = list(repeat(1, len(positive_idxs))) + list(repeat(0, len(negative_idxs)))

    # compute source as the repetition of the current chord over all the
    # targets
    source = list(repeat(self.encode_chord(doc[elem_idx, 0]), len(target)))
    
    return source, target, y, duration

  @staticmethod
  def encode_chord(chord: str) -> np.array:
    """
    Encodes a chord. Need to be overridden to express the right behaviour.

    Args:
        chord (str): Input chord

    Returns:
        np.array: Output representation
    """
    raise NotImplementedError()

  @staticmethod
  def collate_fn(sample: Tuple[List[np.array], List[np.array], np.array, np.array]) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Collate the provided samples in a single batch.

    Args:
        sample (List[List[np.array], List[np.array], np.array]): Input sample

    Returns:
        Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array], np.array, np.array]: Output batch.
    """
    source, target, y, duration = zip(*sample)
    
    source = torch.nn.utils.rnn.pad_sequence(map(torch.tensor, chain(*source)), 
                                             batch_first=True, 
                                             padding_value=0).int()
    
    target = torch.nn.utils.rnn.pad_sequence(map(torch.tensor, chain(*target)), 
                                             batch_first=True, 
                                             padding_value=0).int()
    
    y = torch.tensor(list(chain(*y)))

    duration = torch.tensor(list(chain(*y)))

    return source, target, y, duration


class ChocoChordDataset(ChocoDocumentDataset):
  def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Same as ChocoDocumentDataset but durations are discarded.

    Args:
        idx (int): The item index.

    Returns:
        Tuple[List[int], List[int], List[int], List[float]]: The current item, positive examples, negative examples.
    """
    return super().__getitem__(idx)[:-1]

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
    
    source = torch.tensor((list(chain(*source))))
    target = torch.tensor((list(chain(*target))))
    y = torch.tensor(list(chain(*y)))

    return source, target, y


class ChocoDataModule(pl.LightningDataModule):
  def __init__(self, choco_data_path: str, dataset_cls: ChocoChordDataset, batch_size: int = 1024, context_size: int = 5, negative_sampling_k: int = 20):
    super().__init__()
    self.corpus = ChoCoValidHarteChordsCorpus(choco_data_path)
    self.dataset_cls = dataset_cls
    
    # batch size needs to be adjusted since each sample from the dataset is composed of
    # potentially (context_size * 2) + negative_sampling_k + 1 samples
    self.batch_size = max(1, batch_size // ((context_size * 2) + negative_sampling_k + 1))
    
    self.context_size = context_size
    self.negative_sampling_k = negative_sampling_k

  def prepare_data(self):
    self.dataset = self.dataset_cls(self.corpus, 
                                    context_size=self.context_size, 
                                    negative_sampling_k=self.negative_sampling_k)

  def train_dataloader(self):
    return torch.utils.data.DataLoader(
      self.dataset,
      batch_size=self.batch_size,
      num_workers=os.cpu_count(),
      shuffle=True,
      collate_fn=self.dataset_cls.collate_fn,
      persistent_workers=True,
      prefetch_factor=20
    )
