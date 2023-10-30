from typing import List, Tuple
from itertools import chain
from functools import cache

from pitchclass2vec.data import ChocoChordDataset

class HarteTextDataset(ChocoChordDataset):
  @staticmethod
  def encode_chord(chord: str) -> str:
    """
    Simple textual representation of a chord.

    Args:
        chord (str): 

    Returns:
        str: Chord in Harte format.
    """
    return chord

  def __iter__(self):
    return ([ ann.symbol for ann in doc.annotations ] for doc in self.corpus)