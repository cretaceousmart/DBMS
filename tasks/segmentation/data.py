from typing import List, Tuple

from tqdm import tqdm
import re
import functools
import string
from more_itertools import flatten
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pytorch_lightning as pl

import mirdata
from mir_eval.util import merge_labeled_intervals, adjust_intervals

from harte.harte import Harte

from pitchclass2vec.data import ChocoChordDataset
from pitchclass2vec.pitchclass2vec import Pitchclass2VecModel

SYMBOLS_RE = re.compile("[" + re.escape(string.punctuation) + "]")
NUMBERS_RE = re.compile("[" + re.escape(string.digits) + "]")
CONSECUTIVE_SPACES_RE = re.compile(r"\s+")

VERSE_RE = re.compile(r"(verse)")
PRECHORUS_RE = re.compile(r"(prechorus|pre chorus)")
CHORUS_RE = re.compile(r"(chorus)")
INTRO_RE = re.compile(r"(fadein|fade in|intro)")
OUTRO_RE = re.compile(r"(outro|coda|fadeout|fade-out|ending)")
INSTRUMENTAL_RE = re.compile(r"""(applause|bass|choir|clarinet|drums|flute|harmonica|harpsichord|
                                  instrumental|instrumental break|noise|oboe|organ|piano|rap|
                                  saxophone|solo|spoken|strings|synth|synthesizer|talking|
                                  trumpet|vocal|voice|guitar|saxophone|trumpet)""")
THEME_RE = re.compile(r"(main theme|theme|secondary theme)")
TRANSITION_RE = re.compile(r"(transition|trans)")
OTHER_RE = re.compile(r"(modulation|key change)")


class BillboardDataset(Dataset):
  def __init__(self, pitchclass2vec,test_mode = False):
    super().__init__()
    self.pitchclass2vec = pitchclass2vec
    self.test_mode = test_mode

    billboard = mirdata.initialize('billboard')
    billboard.download()


    # If in Test mode, only load 3 song for testing
    if self.test_mode:
      limit = 5
      test_track_id = billboard.track_ids[0:limit]
      test_tracks = {track_id: billboard.track(track_id) for track_id in test_track_id}
      tracks = test_tracks
    
    # If not in Test mode
    else:
      tracks = billboard.load_tracks()

    self.dataset = list()
    labels = set()

    for i, track in tqdm(tracks.items()):
      try:
        section_intervals = track.named_sections.intervals
        sections = track.named_sections.labels

        # adjust chord intervals to match
        chord_intervals, chords = adjust_intervals(track.chords_full.intervals, 
                                                  labels=track.chords_full.labels, 
                                                  t_min=section_intervals.min(), 
                                                  t_max=section_intervals.max(), 
                                                  start_label="N", 
                                                  end_label="N")

        _, sections, chords = merge_labeled_intervals(section_intervals, sections, chord_intervals, chords)
        preprocessed_labels = [self.preprocess_section(s) for s in sections]
        labels.update(preprocessed_labels)
        self.dataset.append((chords, preprocessed_labels))
      except Exception as e:
        print("Track", i, "not parsable")

    # train the label encoder for each label
    self.label_encoder = OneHotEncoder().fit(np.array(list(labels)).reshape(-1, 1))

  @staticmethod
  def preprocess_section(section: str) -> str:
    """
    Reduce the overall set of sections in few section based on few regex expressions.

    Args:
        section (str): Input section

    Returns:
        str: Unified section
    """
    section = SYMBOLS_RE.sub(" ", section)
    section = NUMBERS_RE.sub(" ", section)
    section = CONSECUTIVE_SPACES_RE.sub(" ", section)

    section = "verse" if VERSE_RE.search(section) else section
    section = "prechorus" if PRECHORUS_RE.search(section) else section
    section = "chorus" if CHORUS_RE.search(section) else section
    section = "intro" if INTRO_RE.search(section) else section
    section = "outro" if OUTRO_RE.search(section) else section
    section = "instrumental" if INSTRUMENTAL_RE.search(section) else section
    section = "theme" if THEME_RE.search(section) else section
    section = "transition" if TRANSITION_RE.search(section) else section
    section = "other" if OTHER_RE.search(section) else section

    section = section.strip()
    return section

  def __len__(self) -> int:
    """
    Returns:
        int: Length of the dataset, defined as the number of chord occurences in the corpus.
    """
    return len(self.dataset)

  @functools.cache
  def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
    """
    Retrieve an item from the dataset.

    Args:
        idx (int): The item index.

    Returns:
        Tuple[np.array, np.array]: The current item and the corresponding labels.
    """
    chords, labels = self.dataset[idx]

    
    embedded_chords = list()
    for c in chords:
      try:
        embedded_chords.append(self.pitchclass2vec[c])
      except:
        embedded_chords.append(self.pitchclass2vec["N"])
    chords = np.array(embedded_chords)
        
    labels = self.label_encoder.transform(np.array(labels).reshape(-1, 1)).toarray()
    return chords, labels 
  @staticmethod
  def collate_fn(sample: Tuple[np.array, np.array]) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Collate the provided samples in a single batch.

    Args:
        sample (Tuple[np.array, np.array]): Input sample

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor]: Output batch.
    """
    chords, labels = zip(*sample)

    mask = [np.ones(x.shape[0]) for x in chords]
    
    chords = pad_sequence(map(torch.tensor, chords),
                          batch_first=True, 
                          padding_value=-1)
    labels = pad_sequence(map(torch.tensor, labels),
                          batch_first=True, 
                          padding_value=0)
    mask = pad_sequence(map(torch.tensor, mask),
                        batch_first=True, 
                        padding_value=0)

    return chords, labels, mask


class SegmentationDataModule(pl.LightningDataModule):
  def __init__(self, dataset_cls: Dataset, pitchclass2vec: Pitchclass2VecModel, batch_size: int = 32, test_size: float = 0.2, valid_size: float = 0.1, test_mode: bool = False):
    """
    Initialize the data module for the segmentation task.

    Args:
        dataset_cls (Dataset): Dataset with segmentation data for training, validation and testing.
        pitchclass2vec (Pitchclass2VecModel): Embedding method.
        batch_size (int, optional): Defaults to 32.
        test_size (float, optional): Defaults to 0.2.
        valid_size (float, optional): Defaults to 0.1.
    """
    super().__init__()
    self.dataset_cls = dataset_cls
    self.batch_size = batch_size
    self.pitchclass2vec = pitchclass2vec
    
    self.test_size = test_size
    self.valid_size = valid_size
    self.train_size = 1 - self.valid_size - self.test_size
    assert self.train_size + self.valid_size + self.test_size == 1.0
    self.test_mode = test_mode
    
  def prepare_data(self):
    """
    Prepare the datasets by splitting data.
    """
    dataset = self.dataset_cls(self.pitchclass2vec, self.test_mode)
    self.train_dataset, self.test_dataset, self.valid_dataset = random_split(
      dataset, 
      [self.train_size, self.test_size, self.valid_size],
      generator=torch.Generator().manual_seed(42))

  def build_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
    """
    Args:
        dataset (Dataset): Dataset used in the dataloader.
        shuffle (bool, optional): Wether the dataloader should shuffle data or not.
          Defaults to True.

    Returns:
        DataLoader: Dataloader built using the specified dataset.
    """
    return torch.utils.data.DataLoader(
      dataset,
      batch_size=self.batch_size,
      num_workers=os.cpu_count(),
      shuffle=shuffle,
      collate_fn=self.dataset_cls.collate_fn,
      persistent_workers=True,
      prefetch_factor=20

    ) 

  def train_dataloader(self) -> DataLoader:
    """
    Returns:
        DataLoader: DataLoader with training data
    """
    return self.build_dataloader(self.train_dataset)

  def val_dataloader(self) -> DataLoader:
    """
    Returns:
        DataLoader: DataLoader with validation data
    """
    return self.build_dataloader(self.valid_dataset, shuffle=False)
    
  def test_dataloader(self) -> DataLoader:
    """
    Returns:
        DataLoader: DataLoader with testing data
    """
    return self.build_dataloader(self.test_dataset, shuffle=False)