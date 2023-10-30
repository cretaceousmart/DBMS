from typing import List
import numpy as np
import os
import pathlib
import json
from harte.harte import Harte

def pitchclass_to_onehot(pitchclass: List[int]) -> np.array:
  """
  Convert a pitchclass representation to a one-hot encoding
  of a pitchclass representation: a 12-dimensional vector
  in which each dimension represents the respective pitchclass.

  Args:
      pitchclass (List[int]): Pitchclass representation

  Returns:
      np.array: One hot encoding.
  """
  return np.eye(12)[pitchclass, :].sum(axis=0).clip(0, 1)

if os.path.exists(pathlib.Path(__file__).parent / "pitchclasses.json"):
    with open(pathlib.Path(__file__).parent / "pitchclasses.json", "r") as f:
        pitchclasses = json.load(f)
else:
    pitchclasses = dict()

def chord_pitchclass(chord: str) -> List[int]:
    """
    Retrieve pitchclass of a chord. Local pre-computed pitchclasses
    are used, if available.

    Args:
        chord (str): Input chord.

    Returns:
        List[int]: Pitchclass of the chord
    """
    if chord not in pitchclasses:
        pitchclasses[chord] = Harte(chord).pitchClasses
    return pitchclasses[chord]