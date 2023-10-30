from typing import Iterable, List, Iterator
import jams
import os
from os import path as osp
import collections
from harte.harte import Harte

ChoCoDocument = collections.namedtuple("ChoCoDocument", ["annotations", "source", "jams"])
HarteAnnotation = collections.namedtuple("HarteAnnotation", ["symbol", "duration"])
HarteSectionAnnotation = collections.namedtuple("HarteSectionAnnotation", ["chord", "section"])

class ChoCoCorpus(object):
  """
  Iterate over all documents contained in ChoCo to retrieve jams files.
  """

  def __init__(self, choco_data_path: str):
    """
    Args:
        choco_data_path (str): Path to ChoCo data directory containing jams files.
    """
    
    #TODO: moodify test jams to jams
    self.data_path = osp.join(choco_data_path, "test_jams")
    


  def _read_jams(self, path: str) -> jams.JAMS:
    """
    Read the jams file from the specified directory. By default validation
    is not enforces.

    Args:
        path (str): Path of the .jams file.

    Returns:
        jams.JAMS: JAMS object
    """
    return jams.load(path, validate=False)

  def __iter__(self) -> Iterator[jams.JAMS]:
    """
    Iterate over all chords progressions.
    Subclass this method to provide custom behaviour.
    By default yields the loaded jams.

    Yields:
        Iterator[jams.JAMS]: Loaded JAMS file.
    """
    
    jams_files = os.walk(self.data_path)

    for (root, _, files) in jams_files:
      for file in files:
        try:
          yield self._read_jams(osp.join(root, file))
        except:
          pass


class ChoCoHarteAnnotationsCorpus(ChoCoCorpus):
  """
  Iterate over all documents contained in ChoCo to retrieve chords in Harte format.
  """
  def _read_jams(self, path: str) -> jams.JAMS:
    """
    Read the jams file from the specified directory. By default validation
    is not enforces.

    Args:
        path (str): Path of the .jams file.

    Returns:
        jams.JAMS: JAMS object
    """
    jam = jams.load(path, validate=False)
    namespaces = [ str(a.namespace) for a in jam.annotations ]
        
    chord_namespace = "chord_harte" if "chord_harte" in namespaces else "chord"
    #BUG fixed: stop using jams.search because the dataset contain time signature
    # annotation = jam.search(namespace=chord_namespace) #Old version
    target_annotation_idx = namespaces.index(chord_namespace)
    annotation = jam.annotations[target_annotation_idx]
    observations = annotation.data

    chords = [HarteAnnotation(obs.value, obs.duration) for obs in observations]
    return ChoCoDocument(chords, source=path, jams=jam)


class ChoCoValidHarteChordsCorpus(ChoCoCorpus):
  """
  Iterate over all documents contained in ChoCo to retrieve valid chords in Harte format.
  """

  def _read_jams(self, path: str) -> jams.JAMS:
    """
    Read the jams file from the specified directory. 
    Raise exceprion if chords are not parsable.

    Args:
        path (str): Path of the .jams file.

    Returns:
        jams.JAMS: JAMS object
    """
    
    jam = jams.load(path, validate=False)
    namespaces = [ str(a.namespace) for a in jam.annotations ]
        
    chord_namespace = "chord_harte" if "chord_harte" in namespaces else "chord"
    # BUG fixed: annotation = jam.search(namespace=chord_namespace) #old version
    target_annotation_idx = namespaces.index(chord_namespace)
    annotation = jam.annotations[target_annotation_idx]
    observations = annotation.data
    chords = [HarteAnnotation(obs.value, obs.duration) for obs in observations]
    
    assert len(chords) > 10
    [Harte(c.symbol) for c in chords]
    return ChoCoDocument(chords, source=path, jams=jam)


class ChoCoIsophonicsHarteCorpus(ChoCoCorpus):
  def __iter__(self) -> Iterator[List[ChoCoDocument]]:
    """
    Yields:
        Iterator[List[HarteAnnotation]]: Progression of chords in Harte format.
    """
    isophonics = [p for p in self.partitions_paths if "isophonics" in p]
    for partition_path in isophonics:
      for jam_path in self._jams_in_partition(partition_path):
        jam = jams.load(jam_path, validate=False)
        namespaces = [ str(a.namespace) for a in jam.annotations ]
        
        chord_namespace = "chord_harte" if "chord_harte" in namespaces else "chord"
        annotation = jam.search(namespace=chord_namespace)
        observations = annotation[0].data
        chords = [HarteAnnotation(obs.value, obs.duration) for obs in observations]
        yield ChoCoDocument(chords, source=jam_path, jams=jam)


class ChoCoHarteAnnotationsSectionCorpus(ChoCoCorpus):
  """
  Iterate over all documents contained in ChoCo to retrieve structural annotation of chords.
  Only available in isophonics partition.
  """
  def __iter__(self) -> Iterator[List[ChoCoDocument]]:
    """
    Yields:
        Iterator[List[HarteSectionAnnotation]]: Progression of chords in Harte format.
    """
    structural_partitions = [p for p in self.partitions_paths if "isophonics" in p]
    for partition_path in structural_partitions:
      for jam_path in self._jams_in_partition(partition_path):
        jam = jams.load(jam_path, validate=False)
        namespaces = [ str(a.namespace) for a in jam.annotations ]
        chord_namespace = "chord_harte" if "chord_harte" in namespaces else "chord"
        structure_namespace = "segment_open"
        
        chord_annotations = jam.search(namespace=chord_namespace)[0]
        structure_annotations = jam.search(namespace=structure_namespace)[0]

        chords = list()
        for structure_obs in structure_annotations.data:
          section = structure_obs.value
          start = structure_obs.time
          duration = structure_obs.duration
          end = start + duration

          chords.extend(HarteSectionAnnotation(HarteAnnotation(obs.value, obs.duration), section) 
                        for obs in chord_annotations.slice(start, end))

        yield ChoCoDocument(chords, source=jam_path, jams=jam)


if __name__ == "__main__":
  # harte_corpus = ChoCoHarteAnnotationsSectionCorpus("/home/n28div/university/thesis/choco")
  harte_corpus = ChoCoHarteAnnotationsSectionCorpus("/app/choco")
  print(next(iter(harte_corpus)))


#def export_corpus(choco_path: str, outfile: str):
#  corpus = ChoCoHarteCorpus(choco_path)
#
#  with open(outfile, "w") as f:
#    for chords in corpus:
#      progression = "|".join(chords) + "\n"
#      f.write(progression)