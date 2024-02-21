# TODO: 'raw_detections' - Load all raw classifier detections << Needed in future for evaluating alternate classifiers one-to-one, or to use non-species-specific detections for evaluation

# Declare some configuration variables ------------------------------------

# Root directory for the annotations data
dir_annotations = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020'
# DEBUG:
dir_annotations = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment4/SMA00424_20200521'


# Root directory for the raw detection data
dir_detections = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/raw_detections/2020'

species_to_evaluate = 'all' # Evaluate all species classes...
# species_to_evaluate = ['american robin', 'band-tailed pigeon', 'barred owl', 'american kestrel'] # ...or look at just a few

plot = False # Plot the results

# Load required packages -------------------------------------------------
import os                       # File navigation
import pandas as pd             # Data manipulation
import matplotlib.pyplot as plt # Plotting
import sklearn.metrics          # Classifier evaluation
import re                       # Regular expressions
import sys

# Define utility functions -------------------------------------------------
# These helper functions are used to easily encapsulate repeated tasks

# Print warning
def print_warning(l):
    print(f'\033[33mWARNING: {l}\033[0m')

# Function to parse species name, confidence, serial number, date, and time from filename
def parse_metadata_from_file(filename):

    # Regular expression pattern to match the filename
    pattern = r'^(.+)-([\d.]+)_(\w+)_(\d{8})_(\d{6})\.(\w+)$'
    match = re.match(pattern, filename)

    if match:
        species = match.group(1)
        confidence = float(match.group(2))
        serialno = match.group(3)
        date = match.group(4)
        time = match.group(5)
        return species, confidence, serialno, date, time
    else:
        print_warning("Unable to parse info from filename:", filename)
        return

# Find all selection table files under a root directory
def find_files(directory, filetype):

    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(filetype):
                results.append(os.path.join(root, file))
    return results

# Declare 'evaluated_detections' - Get the list of files (detections) that were actually evaluated by annotators
# These are all the .wav files under each site-date folder in e.g. 'annotation/data/_annotator/2020/Deployment4/SMA00410_20200523'
evaluated_detection_files_paths = find_files(dir_annotations, '.wav')
evaluated_detection_files = list(map(os.path.basename, evaluated_detection_files_paths))
# Parse the species names and confidence scores from the filenames
file_metadata = list(map(lambda f: parse_metadata_from_file(f), evaluated_detection_files))
species, confidences, serialnos, dates, times = zip(*file_metadata)

evaluated_detections = pd.DataFrame({
    'file': evaluated_detection_files,
    'species_predicted': species,
    'conf': confidences,
    'date': dates,
    'time': times
})
evaluated_detections['species_predicted'] = evaluated_detections['species_predicted'].str.lower()

# Declare 'raw_annotations' - Load, aggregate, and clean all annotation labels (selection tables), which is a subset of 'evaluated_detections'
# Load selection table files and combine into a single 'raw_annotations' dataframe
selection_tables = find_files(dir_annotations, '.txt')
raw_annotations = pd.DataFrame()
for table_file in sorted(selection_tables):
    print(f'Loading file {os.path.basename(table_file)}...')

    table = pd.read_csv(table_file, sep='\t') # Load the file as a dataframe

    # Clean up data by normalizing column names and species values to lowercase
    table.columns = table.columns.str.lower()
    table['species'] = table['species'].str.lower()

    # Check the validity of the table
    cols_needed = ['species', 'begin file', 'file offset (s)', 'delta time (s)']
    if not all(col in table.columns for col in cols_needed):
        missing_columns = [col for col in cols_needed if col not in table.columns]
        print_warning(f"{os.path.basename(table_file)} is missing columns {missing_columns}")
        continue
    if table.empty:
        print_warning(f'{table_file} has no selections. Continuing...')
        continue

    table = table[cols_needed] # Discard unnecessary data

    if table.isna().any().any():
        print_warning(f'{table_file} contains NaN values')
        # table = table.dropna()

    # Parse the species names and confidence scores from 'Begin File' column
    file_metadata = list(map(lambda f: parse_metadata_from_file(f), table['begin file']))
    species, confidences, serialnos, dates, times = zip(*file_metadata)

    table.rename(columns={'species': 'label_truth'}, inplace=True) # Rename 'species' to 'label_truth'
    table.insert(0, 'species_predicted', species) # Add column for species predicted by the classifier
    table.insert(2, 'confidence', confidences) # Add column for confidence values
    table['serialno'] = serialnos

    # Clean up species names
    table['species_predicted'] = table['species_predicted'].str.lower()
    table['species_predicted'] = table['species_predicted'].str.replace('_', "'")
    table['label_truth'] = table['label_truth'].str.replace('_', "'")

    table.rename(columns={'begin file': 'file'}, inplace=True) # Rename 'begin file' to 'file'

    raw_annotations = pd.concat([raw_annotations, table], ignore_index=True) # Store the table

# More clean up of typos
raw_annotations.loc[raw_annotations['label_truth'].str.contains('not|non', case=False), 'label_truth'] = '0' # 0 indicates the species_predicted is NOT present
raw_annotations['label_truth'].replace(['unknown', 'unkown'], 'unknown', inplace=True)

# # Exclude files with an "unknown" label_truth from consideration
# # TODO: remove unknowns from evaluated_detections as well!
# unknown_evaluated_detections = raw_annotations.groupby('file').filter(lambda x: 'unknown' in x['label_truth'].values)
# print(unknown_evaluated_detections.to_string())
# sys.exit()
# raw_annotations = raw_annotations.groupby('file').filter(lambda x: 'unknown' not in x['label_truth'].values)

# print(raw_annotations)

unique_labels = sorted(raw_annotations['label_truth'].unique()) # Get a sorted list of unique species names
print(f'{len(unique_labels)} unique labels annotated: {unique_labels}')

# TODO: Throw warning if missing annotations for any species in species_predicted
evaluated_species = sorted(raw_annotations['species_predicted'].unique())
species_classes = pd.read_csv('/Users/giojacuzzi/repos/olympic-songbirds/classification/species_list/species_list_OESF.txt', header=None) # Get list of all species
species_classes = [name.split('_')[1].lower() for name in species_classes[0]]

missing_species = [species for species in species_classes if species not in evaluated_species]
if (len(missing_species) > 0):
    print_warning(f'Missing annotations for the following species:\n{missing_species}')

# Evaluate model performance on species classes -------------------------------------------------
# This section uses the annotation data to evaluate performance metrics for each species class,
# namely the precision, recall, precision-recall curve and AUC.

# If evaluating all species classes, retrieve their names from the data
if species_to_evaluate == 'all':
    species_to_evaluate = sorted(raw_annotations['species_predicted'].unique())

# For each unique species (or a particular species) in evaluated_detections
annotations = pd.DataFrame()
for species_name in species_to_evaluate:
    print(f'Evaluating class "{species_name}"...')

    # Declare a dataframe of annotations for that species 'annotations_species'
    annotations_species = raw_annotations[raw_annotations['species_predicted'] == species_name]
    # TODO: If there is no selection table for the species at a given site, throw a warning

    # TODO: In the future, include examples from all other species detections as well. This requires pulling confidence scores from the raw data,
    # rather than the detection annotations

    # For each unique file (i.e. detection), determine if the species was present (1) or not (0) among all labels
    labels_binary = annotations_species.groupby('file').apply(lambda group: 1 if species_name in group['label_truth'].values else 'unknown' if 'unknown' in group['label_truth'].values else 0).reset_index()
    labels_binary.columns = ['file', 'label_truth']
    labels_binary = pd.merge(
        labels_binary,
        annotations_species[['file', 'confidence']].drop_duplicates(), # Get the detection confidence associated with this file
        on=['file'],
        how='left'
    )
    print(labels_binary)

    annotations = pd.concat([annotations, labels_binary], ignore_index=True) # Store the annotations

    # For each detection (file) of that species in evaluated_detections

        # Determine the detection's confidence score. For now, do this by pulling it from the file.
        # In the future, instead reference the associated raw detection(s) for the correct score(s). 

        # Match any associated annotations from 'annotations' for the detection.
        # Aggregate these annotations (1 for present in file, 0 for not present in file).
        # If there are no labels of the species, it is not present (0).

        # Save the detection's species, time, confidence, and label_truth to 'annotations_species'

    # Save 'annotations_species' to 'annotations'

print('annotations!=================================')
print(annotations.to_string())

print('evaluated detections!========================')
print(evaluated_detections.to_string())

print('merge========================================')
detection_labels = pd.merge(
    evaluated_detections,
    annotations,
    on=['file'],
    how='left' # 'left' to include all detections (once annotations are complete), 'inner' to only include those detections that have annotations
)
detection_labels = detection_labels.sort_values(by=['species_predicted', 'conf'])

if (len(missing_species) > 0):
    print_warning('Removing species with missing annotations')
    detection_labels = detection_labels[~detection_labels['species_predicted'].isin(missing_species)]

if detection_labels['label_truth'].isna().any():
    print_warning('Assuming missing (NaN) label_truth values due to lack of annotation selections are false positives (0)...')
    detection_labels['label_truth'] = detection_labels['label_truth'].fillna(0)
print(detection_labels.to_string())

# TODO: Evaluate performance after removing rows with 'unknown' label_truth values