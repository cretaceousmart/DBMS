from harte.harte import Harte
import jams
import collections
import os
from tqdm import tqdm 
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message=".*is not of type 'string'")

ChoCoDocument = collections.namedtuple("ChoCoDocument", ["annotations", "source", "jams"])
HarteAnnotation = collections.namedtuple("HarteAnnotation", ["symbol", "duration"])
HarteSectionAnnotation = collections.namedtuple("HarteSectionAnnotation", ["chord", "section"])


# Those are the 24 basic chord that define by madmom
basic_chords = {'C:maj','C#:maj','D:maj','D#:maj','E:maj','F:maj','F#:maj','G:maj','G#:maj','A:maj','A#:maj','B:maj',

'C:min','C#:min','D:min','D#:min','E:min','F:min','F#:min','G:min','G#:min','A:min','A#:min','B:min',

'N'}

def basic_root(root_name):
    """
    Convert roots with double sharps or flats to their enharmonic equivalents.
    E.g., 'Fb' to 'E', 'C##' to 'D', 'B#' to 'C', 'E#' to 'F', etc.
    """
    # Define the mapping of double sharps and flats to their enharmonic equivalents
    enharmonic_equivalents = {
        'Fb': 'E', 'E#': 'F', 'B#': 'C', 'Cb': 'B',
        'C##': 'D', 'D##': 'E', 'E##': 'F#', 'F##': 'G',
        'G##': 'A', 'A##': 'B', 'B##': 'C#', 'Bb': 'A#',
        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#',
        'Bbb': 'A', 'Abb': 'G', 'Gbb': 'F', 'Fbb': 'Eb',
        'Ebb': 'D', 'Dbb': 'C', 'Cbb': 'Bb'
    }
    
    # Return the enharmonic equivalent if it exists, otherwise return the root as is
    return enharmonic_equivalents.get(root_name, root_name)


def map_to_basic_chords(chord):
    # Handle special markers
    if chord in ['N', 'X']:
        return 'N'
    
    # Clean up uncommon markings in the chord
    chord = chord.replace("/bb1", "")
    
    try:
        # Get the root note
        root = Harte(chord).get_root()
        # Convert to the standard root if necessary
        root = basic_root(root)
        
        # Check if the chord contains 'maj' or 'min'
        if 'maj' in chord:
            return f'{root}:maj'
        elif 'min' in chord:
            return f'{root}:min'
        else:
            if 'b3' in chord or 'b10' in chord:
                return f'{root}:min'
            elif '3' in chord or '10' in chord:
                return f'{root}:maj'
            elif 'sus' in chord or 'dim' in chord or 'aug' in chord:
                return 'N'
            else:
                return f'{root}:maj'
    except Exception as e:
        # Handle any exceptions that occur
        print(f"Error processing chord {chord}: {e}")
        return 'N'  # Return 'N' for unprocessable chords






def update_jams_with_basic_chords(file_path):
    jam = jams.load(file_path, validate=False)
    
    # Need to delete all the annotation unless it's namespace is chord_harte or chord
    annotations_to_keep = jams.AnnotationArray()

    for ann in jam.annotations:
        if ann.namespace in ['chord_harte', 'chord']:
            annotations_to_keep.append(ann)
    jam.annotations = annotations_to_keep
    try:
        # Find the index of the chord_harte or chord annotation
        chord_namespace = "chord_harte" if any(ann.namespace == "chord_harte" for ann in jam.annotations) else "chord"
        target_annotation_idxs = [i for i, ann in enumerate(jam.annotations) if ann.namespace in ['chord_harte', 'chord']]
        
        if len(target_annotation_idxs) == 0:
            raise ValueError(f"No chord annotation found in {file_path}")

        for idx in target_annotation_idxs:

            # If the target annotation is found, process it
            if idx is not None:
                annotation = jam.annotations[idx]
                new_data = []
                for obs in annotation.data:
                    # Ensure the duration is non-negative
                    obs_duration = max(obs.duration, -1 * obs.duration)
                    new_value = map_to_basic_chords(obs.value)
                    new_obs = jams.Observation(time=obs.time,
                                            duration=obs_duration,
                                            value=new_value,
                                            confidence=obs.confidence)
                    new_data.append(new_obs)
                
                # Create a new Annotation object and replace the previous annotation
                new_annotation = jams.Annotation(namespace=chord_namespace)
                new_annotation.data = new_data
                jam.annotations[idx] = new_annotation
        
        # Save the updated JAMS file
        jam.save(file_path,strict=False)

    except Exception as e:
        # If an error occurs, print the file name and error message
        print([ann.namespace for ann in jam.annotations])
        print(f"Error processing file {file_path}: {e}")
        raise

folder_path = r"D:\Dev\chord-embeddings\app\choco_dataset\v1.0.0\jams_in_24_chords"
all_files = [f for f in os.listdir(folder_path) if f.endswith('.jams')]
all_files.sort()  # Sort the files if not already sorted

# Only iterate over the unprocessed files starting from start_index
start_index = 9430
for filename in tqdm(all_files[start_index:], initial=start_index, total=len(all_files), desc="Processing JAMS files"):
# for filename in tqdm(all_files, desc="Processing JAMS files"):
    file_path = os.path.join(folder_path, filename)
    update_jams_with_basic_chords(file_path)

warnings.filterwarnings('default', category=UserWarning, message=".*is not of type 'string'")