from typing import Dict
import os
import pickle
import numpy as np

from pitchclass2vec.encoding.rdf import RDFChordsDataset
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker, WLWalker, HALKWalker


class BaseRdf2VecModel(object):
  WALKER = None

  def __init__(self, max_depth: int, max_walks: None, max_epochs: int = 1, **kwargs):
    """
    Initialize a random walk based rdf2vec.

    Args:
        max_depth (int): Maximum depth in walking the graph.
        max_walks (None): Maximum number of walks.
        max_epochs (int, optional): Number of training epochs. Defaults to 1.
    """
    self.transformer = RDF2VecTransformer(
      Word2Vec(epochs=max_epochs),
      walkers=[self.WALKER(max_depth, None if max_walks < 0 else max_walks, with_reverse=True)],
      verbose=1
    )
    self.embeddings = None
    self.embedding_dim = 100

  def train(self, kg: RDFChordsDataset):
    """
    Train the rdf2vec method. 
    Out of vocabulary terms are embedded as tandom vectors.

    Args:
        kg (RDFChordsDataset): Input rdf dataset.
    """
    embeddings, _ = self.transformer.fit_transform(kg.kg, list(kg.vocab.values()))
    self.embeddings = {
      chord: emb
      for chord, emb in zip(kg.vocab.keys(), embeddings)
    }
    self.embeddings["UNK"] = np.random.rand(len(embeddings[0]))

  def __getitem__(self, key: str) -> np.array:
    """
    Returns the embedding of a chord.

    Args:
        key (str): Input chord.

    Returns:
        np.array: Embedding of that chord.
    """
    return self.embeddings[key] if key in self.embeddings else self.embeddings["UNK"]

  def save(self, path: str):
    """
    Save the rdf2vec model to disk.

    Args:
        path (str): Path in which the model is saved to.
    """
    with open(path, "wb") as f:
      pickle.dump(self, f)

  @staticmethod
  def load_from_checkpoint(path: str):
    """
    Load an rdf2vec object.

    Args:
        path (str): Path containing the model.

    Returns:
        rdf2vec object
    """
    with open(path, "rb") as f:
      obj = pickle.load(f)
    return obj


class RandomWalkRdf2VecModel(BaseRdf2VecModel):
  WALKER = RandomWalker


class WLWalkRdf2VecModel(BaseRdf2VecModel):
  WALKER = WLWalker


class HALKWalkRdf2VecModel(BaseRdf2VecModel):
  WALKER = HALKWalker
