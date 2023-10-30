from typing import List
from suffix_tree import Tree
import numpy as np

def FORM(seq: List[str]) -> np.array:
  """
  Implement the FORM[1] algorithm to perform structure segmentation
  using symbolic representation of chords.

  Args:
      seq (List[str]): Sequence of chords

  Returns:
      np.array: 
        Segmentation of the piece where each chord as been labeled 
        with the corresponding section.
  """
  # Note: Chord simplification measure is relaxed
  tree = Tree({ "": seq })
  tree.root.compute_left_diverse()

  # prune non-left-diverse nodes
  def prune_children(node):
    node.children = { k: v for k, v in node.children.items() if v.is_left_diverse }
  tree.root.pre_order(prune_children)

  # get all root subtrees
  subtrees = list()
  def get_subtrees(node):
    if len(node.children) == 0:
      subtrees.append(node.path.seq)
  tree.root.post_order(get_subtrees)

  # egment the chord sequence at every position where a repeated pattern starts
  # and a subtree label change
  segmentation = np.full_like(seq, -1, dtype=float)
  seq += [None]
  all_candidates = list(range(len(subtrees)))
  candidates = all_candidates
  i = 0
  j = 0
  while i < len(seq):
    chord = seq[i]
    # keep all subtrees labels that match current chord
    valid_candidates = [c for c in candidates 
                        if j < len(subtrees[c]) and subtrees[c][j] == chord]
    if len(valid_candidates) == 0:
      if len(candidates) == 1 and j == len(subtrees[candidates[0]]):
        # a label switched: set it to the segmentation and begin
        # a new search
        label = candidates[0]
        segmentation[i - j: i] = label
        candidates = all_candidates
        j = 0
      else:
        # no candidates could be continued from previous strike
        # search for new candidates
        j = 0
        candidates = [c for c in all_candidates 
                      if j < len(subtrees[c]) and subtrees[c][j] == chord]
    else:
      # advance candidate search
      candidates = valid_candidates
      j += 1
    i += 1

  # for those section which have not been segmented the preious section is extended
  not_segmented_idxs = np.where(segmentation == -1.0)[0]
  for idx in not_segmented_idxs:
    if idx > 0:
      segmentation[idx] = segmentation[idx - 1]
  # special handle index 0 -> perform all propagation and if it doens't have a label
  # extend the next section backwards
  if segmentation[0] == -1: segmentation[0] = segmentation[1]

  return segmentation