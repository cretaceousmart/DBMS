from typing import List, Tuple, Union

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
from pitchclass2vec.pitchclass2vec import NaiveEmbeddingModel

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
  def __init__(self, embedding_model,test_mode = False, full_chord = True):
    super().__init__()
    self.embedding_model = embedding_model
    self.test_mode = test_mode
    self.full_chord = full_chord

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
        if self.full_chord:
          chord_intervals, chords = adjust_intervals(track.chords_full.intervals, 
                                                    labels=track.chords_full.labels, 
                                                    t_min=section_intervals.min(), 
                                                    t_max=section_intervals.max(), 
                                                    start_label="N", 
                                                    end_label="N")
        else:
          chord_intervals, chords = adjust_intervals(track.chords_majmin.intervals, 
                                                    labels=track.chords_majmin.labels, 
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

  # @functools.cache
  def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:

    # TODO: change output from (chord, labels, mask) to (source, target, source_mask, target_mask)
    """
    Retrieve an item from the dataset. Not in a batch
    In Billboard dataset:
    {'bridge': 1, 'chorus': 2, 'instrumental': 3, 'interlude': 4, 'intro': 5, 'other': 6, 'outro': 7, 'refrain': 8, 'theme': 9, 'transition': 10, 'verse': 11}

    Args:
        idx (int): The item index.

    Returns:
        Tuple[np.array, np.array]: The current item and the corresponding labels.
    """
    chords, labels = self.dataset[idx]


    # Encode each chord into a 3-dimensional vector
    embedded_chords = list()
    for c in chords:
      try:
        embedded_chords.append(self.embedding_model[c])    
      except:
        embedded_chords.append(self.embedding_model["N"])
    chords = np.array(embedded_chords)
    

    # Encode the label into one-hot
    
    # label_to_int = {'bridge': 0, 'chorus': 1, 'instrumental': 2, 'interlude': 3, 'intro': 4, 'other': 5, 'outro': 6, 'refrain': 7, 'theme': 8, 'transition': 9, 'verse': 10}

    # Here the number must be consistent with the padding value in: labels = pad_sequence(map(torch.tensor, labels), batch_first=True, padding_value=0)
    label_to_int = {
      '<PAD>': 0,
      '<SOS>': 1,
      '<EOS>': 13,
      'bridge': 2,
      'chorus': 3,
      'instrumental': 4,
      'interlude': 5,
      'intro': 6,
      'other': 7,
      'outro': 8,
      'refrain': 9,
      'theme': 10,
      'transition': 11,
      'verse': 12
    }
    int_to_label = {idx: label for label, idx in label_to_int.items()}

    labels_set = set(i for i in label_to_int.keys())

    int_labels = [label_to_int['<SOS>']] + [label_to_int[e] for e in labels] + [label_to_int['<EOS>']]

    one_hot_labels = torch.nn.functional.one_hot(torch.tensor(int_labels), num_classes=len(labels_set)).cpu().detach().numpy().astype(np.float64)

    return chords, one_hot_labels
  
  @staticmethod
  def collate_fn(batch: List[Tuple[np.array, np.array]]) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    chords, labels = zip(*batch)

    # 将 numpy 数组转换为 torch 张量，并进行填充以确保批次中所有样本的一致性
    chords = pad_sequence(map(torch.tensor, chords), batch_first=True, padding_value=-1).float()
    labels = pad_sequence(map(torch.tensor, labels), batch_first=True, padding_value=0)

    source_mask = (chords != -1)

    trg_lens = torch.sum(labels != 0, dim=1)
    max_len = trg_lens.max()
    target_mask = (torch.arange(max_len).expand(len(labels), max_len) < trg_lens.unsqueeze(1)).bool()

    # Return: chord is source sequence, labels is target sequence
    return chords, labels, source_mask, target_mask



class SegmentationDataModule(pl.LightningDataModule):
  def __init__(self, 
               dataset_cls: Dataset, 
               embedding_model: Union[Pitchclass2VecModel, NaiveEmbeddingModel], 
               batch_size: int = 32, test_size: float = 0.2, valid_size: float = 0.1, 
               test_mode: bool = False,
               full_chord: bool = False):
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
    self.embedding_model = embedding_model
    
    self.test_size = test_size
    self.valid_size = valid_size
    self.train_size = 1 - self.valid_size - self.test_size
    assert self.train_size + self.valid_size + self.test_size == 1.0
    self.test_mode = test_mode
    self.full_chord = full_chord
    
  def prepare_data(self):
    """
    Prepare the datasets by splitting data.
    """
    dataset = self.dataset_cls(self.embedding_model, self.test_mode, self.full_chord)

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
  



"""
-----------------------------------------
Test Code:
-----------------------------------------
from tasks.segmentation.data import BillboardDataset, SegmentationDataModule
from pitchclass2vec import encoding, model
from pitchclass2vec.pitchclass2vec import NaiveEmbeddingModel

encoder = encoding.RootIntervalDataset
embedding_model = NaiveEmbeddingModel(
                                encoding_model=encoder, 
                                embedding_dim=3, # dim=3 because each '24 basic chords' only contain 3 notes
                                norm=False)

data = SegmentationDataModule(  dataset_cls=BillboardDataset, 
                                    embedding_model=embedding_model, 
                                    batch_size = 5,
                                    test_mode = False,
                                    full_chord = False
                                    )

                                    
data.prepare_data()
train_loader = data.train_dataloader()

for i, batch in enumerate(train_loader):
    
    chord, labels, mask = batch  # 添加了 mask

    print(f"Batch {i}:")
    print(f"Chord size: {chord.size()}")
    print(f"Labels size: {labels.size()}")
    print(f"mask size: {mask.size()}")

    if i == 0:
        break
-----------------------------------------
Output:
-----------------------------------------
Chord size: torch.Size([5, 273, 3])
Labels size: torch.Size([5, 273, 11])
mask size: torch.Size([5, 273])
"""