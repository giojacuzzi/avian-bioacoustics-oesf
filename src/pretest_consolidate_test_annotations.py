# Consolidate test annotation data into a single csv with columns:
# source - the source of this sample, i.e. the original label assigned to the example by the pre-trained model
# file - the file of the example
# path - the full path to the example
# labels - the true labels for the example, separated by token ', '

# Directory to consolidate containing both annotation .txt and raw audio .wav data
in_dir = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/transfer learning/data/test/2020'

import pandas as pd
import matplotlib.pyplot as plt
from annotation.annotations import *
from utils.log import *
from classification.performance import *
from utils.files import *
import os
import re

# Retrieve and validate raw annotation data
print('Retrieving annotation data...')
raw_annotations = get_raw_annotations(dirs = [in_dir], overwrite = False)

# Check for missing annotations
species = labels.get_species_classes()
species_to_evaluate = sorted(species)
# Warn if missing annotations for any species in label_predicted
species_annotated = [label for label in sorted(raw_annotations['label_truth'].astype(str).unique()) if label in species]
print(f'Retrieved {len(raw_annotations)} total annotations for {len(species_annotated)}/{len(species)} species classes')
if len(species_annotated) < len(species):
    print_warning(f'Missing positive annotations for species: {[label for label in species if label not in species_annotated]}')
print(raw_annotations)

print('Finding audio data paths...')
audio_paths = []
for root, dirs, files in os.walk(in_dir):
    for example_file in files:
        if example_file.endswith(".wav"):
            audio_paths.append(os.path.join(root, example_file))

print(f'Found {len(audio_paths)} raw audio files')
print(f"Found annotations for {len(raw_annotations['file'].unique())} audio files")

# Load class labels
class_labels_csv_path = os.path.abspath(f'data/class_labels.csv')
class_labels = pd.read_csv(class_labels_csv_path)

# Create a dataframe consisting of 1 row for each example and the following columns:
# target = the source of the audio data, i.e. the original label assigned by the pre-trained model
# file = the audio data file basename, including extension
# path = the full path to the audio data file
# labels = the true labels for the example
# dataset = 'test'

test_data_annotations = pd.DataFrame(columns=['target', 'file', 'path', 'labels', 'dataset'])
for example_audio_path in audio_paths:

    # print("========================================================")
    example_file = os.path.basename(example_audio_path)

    example_annotations = raw_annotations[raw_annotations['file'] == example_file]
    example_annotations.fillna('', inplace=True)
    # print(example_annotations)


    pattern = r"^(.*?)-(-?\d+\.\d+).*$" # all text before a hyphen followed by a decimal number
    match = re.match(pattern, example_file)
    if match:
        target = match.group(1).lower()
    else:
        print(f'ERROR: Could not get target from file {example_file}')
        sys.exit()
    # print(f"target: {target}")

    # print(f"example_file: {example_file}")
    # print(f"example_audio_path: {example_audio_path}")

    # Create a mapping dictionary from 'label' to 'label_birdnet'
    label_mapping = pd.Series(class_labels['label_birdnet'].values, index=class_labels['label']).to_dict()

    target_label_birdnet = label_mapping.get(target, target)
    # print(f"target_label_birdnet: {target_label_birdnet}")

    # Get unique true labels
    unique_labels = sorted(example_annotations['label_truth'].unique())
    # unique_labels = [f'-{target_label_birdnet}' if l == 'not_target' else l for l in unique_labels] # replace 'not_target' with negative label
    # print(f"unique_labels: {unique_labels}")

    # Convert to string with ', ' delimiter
    example_labels = ', '.join([label_mapping.get(l, l) for l in unique_labels])
    # print(f"example_labels: {example_labels}")

    row = {
        'target':  target_label_birdnet,
        'file':    example_file,
        'path':    example_audio_path,
        'labels':  example_labels,
        'dataset': 'test'
    }
    test_data_annotations = pd.concat([test_data_annotations, pd.DataFrame([row])], ignore_index=True)

out_path = 'data/testing/testing_data_annotations.csv'
print('Sorting results...')
test_data_annotations = test_data_annotations.sort_values(by="target", ascending=True)
test_data_annotations.to_csv(out_path)
print(f'Saved consolidated test annotation data to {out_path}')
