from typing import List, Tuple, Dict
import os
from collections import Counter
from itertools import chain, repeat
import pathlib
import pickle
from functools import cache

from rdflib import RDF, RDFS, URIRef, Graph
from pyrdf2vec.graphs import KG

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm

CACHE_PATH = pathlib.Path(__file__).parent.parent / ".cache"
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)


class RDFChordsDataset(torch.utils.data.Dataset):
  def __init__(self, kg_path: str):
    """
    Args:
        kg_path (str): Path to the knowledge graph with chords represented using the chord ontology.
    """
    super().__init__()
    self._cache_file = CACHE_PATH / "kg.pickle"

    if os.path.exists(self._cache_file):
      with open(self._cache_file, "rb") as f:
        self.kg, self.graph = pickle.load(f)
    else:
      # load the KG graph and extract all the chords
      self.graph = Graph()
      self.graph.parse(kg_path)
      self.kg = KG(kg_path)

      with open(self._cache_file, "wb") as f:
        pickle.dump((self.kg, self.graph), f)

    self.vocab = dict()
    chord_IRI = URIRef("http://purl.org/ontology/chord/Chord")
    for chord, _, _ in self.graph.triples((None, RDF.type, chord_IRI)):
      label = str(self.graph.value(chord, RDFS.label))
      self.vocab[label] = str(chord)

  @staticmethod
  def encode_chord(chord: str) -> str:
    """
    Args:
        chord (str): Input chord

    Returns:
        str: The input chord.
    """
    return chord