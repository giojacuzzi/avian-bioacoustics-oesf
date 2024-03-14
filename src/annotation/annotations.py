##
# Validate evaluation data and evaluate classifier performance for each species class
# Requires annotations from Google Drive. Ensure these are downloaded locally before running.

# Load required packages -------------------------------------------------
import sys
import os                       # File navigation
import pandas as pd             # Data manipulation
from utils import files, labels
from utils.log import *

# Load and collate all annotations from the given directories
# Directories must contain both .txt Raven Pro selection tables and their corresponding .wav detection files
# Also performs validation
#
# Returns a dataframe of collated and validated annotations from all Raven Pro selection tables under the given directory
def collate_annotations(dirs, overwrite=False, print_annotations=False):

    annotations_filepath = 'data/annotations/processed/annotations_processed.csv'

    # Return previously processed annotations unless overwrite requested
    if os.path.exists(annotations_filepath) and not overwrite:
        print(f'Retrieving annotations from {annotations_filepath}')
        return pd.read_csv('data/annotations/processed/annotations_processed.csv', index_col=None)

    # Declare 'evaluated_detections' - Get the list of files (detections) that were actually evaluated by annotators
    # These are all the .wav files under each site-date folder in e.g. 'annotation/data/_annotator/2020/Deployment4/SMA00410_20200523'
    evaluated_detection_files_paths = []
    for dir in dirs:
        evaluated_detection_files_paths.extend(files.find_files(dir, '.wav'))

    if len(evaluated_detection_files_paths) == 0:
        print_error('No .wav detection files found')
        exit()

    evaluated_detection_files = list(map(os.path.basename, evaluated_detection_files_paths))
    # Parse the species names and confidence scores from the .wav filenames
    file_metadata = list(map(lambda f: files.parse_metadata_from_file(f), evaluated_detection_files))
    species, confidences, serialnos, dates, times = zip(*file_metadata)

    evaluated_detections = pd.DataFrame({
        'file': evaluated_detection_files,
        'species_predicted': species,
        'confidence': confidences,
        'date': dates,
        'time': times,
        'serialno': serialnos
    })
    evaluated_detections['species_predicted'] = evaluated_detections['species_predicted'].str.lower()

    # Declare 'raw_annotations' - Load, aggregate, and clean all annotation labels (selection tables), which is a subset of 'evaluated_detections'
    # Load selection table files and combine into a single 'raw_annotations' dataframe
    selection_tables = []
    for dir in dirs:
        selection_tables.extend(files.find_files(dir, '.txt'))

    if len(selection_tables) == 0:
        print_error('No Raven Pro .txt selection table files found')
        exit()

    raw_annotations = pd.DataFrame()
    print('Loading annotation selection tables...')
    for table_file in sorted(selection_tables):
        # print(f'Loading file {os.path.basename(table_file)}...')

        table = pd.read_csv(table_file, sep='\t') # Load the file as a dataframe

        # Clean up data by normalizing column names and species values to lowercase
        table.columns = table.columns.str.lower()
        table['species'] = table['species'].astype(str).str.lower()

        # Check the validity of the table
        cols_needed = ['species', 'begin file', 'file offset (s)', 'delta time (s)']
        if not all(col in table.columns for col in cols_needed):
            missing_columns = [col for col in cols_needed if col not in table.columns]
            print_warning(f"Missing columns {missing_columns} in {os.path.basename(table_file)}. Skipping...")
            continue

        if table.empty:
            print_warning(f'{table_file} has no selections.')
            continue

        table = table[cols_needed] # Discard unnecessary data

        if table.isna().any().any():
            print_warning(f'{table_file} contains NaN values')

        # Parse the species names and confidence scores from 'Begin File' column
        file_metadata = list(map(lambda f: files.parse_metadata_from_file(f), table['begin file']))
        species, confidences, serialnos, dates, times = zip(*file_metadata)

        table.rename(columns={'species': 'label_truth'}, inplace=True) # Rename 'species' to 'label_truth'
        table.insert(0, 'species_predicted', species) # Add column for species predicted by the classifier
        table.insert(2, 'confidence', confidences) # Add column for confidence values
        # table['serialno'] = serialnos

        # Clean up species names
        table['species_predicted'] = table['species_predicted'].str.lower()
        table['species_predicted'] = table['species_predicted'].str.replace('_', "'")
        table['label_truth'] = table['label_truth'].str.replace('_', "'")

        table.rename(columns={'begin file': 'file'}, inplace=True) # Rename 'begin file' to 'file'

        if (table['label_truth'] == 'nan').any() or table['label_truth'].isnull().any():
            print_warning(f'{table_file} contains annotations with missing labels')

        raw_annotations = pd.concat([raw_annotations, table], ignore_index=True) # Store the table

    # More clean up of typos
    print('Cleaning annotation labels...')
    raw_annotations['label_truth'] = raw_annotations['label_truth'].apply(labels.clean_label)

    # Warn if any annotations extend beyond the length of a single detection file
    for i, row in raw_annotations.iterrows():
        if (row['file offset (s)'] + row['delta time (s)']) > 3.0:
            print_warning(f'Annotation extends beyond detection length: {row["file"]}')

    print(raw_annotations)
    # Include any evaluated detections for the species without annotations (which should all be FP)
    empty_annotations = evaluated_detections.merge(raw_annotations.drop(columns='species_predicted'), on=['file', 'confidence'], how='outer', indicator=True)
    print(empty_annotations)
    empty_annotations = empty_annotations[empty_annotations['_merge'] != 'both'] # Filter out the rows where the indicator value is 'both'
    print(empty_annotations)
    empty_annotations = empty_annotations.drop(columns='_merge')
    print(empty_annotations)
    if not empty_annotations.empty:
        print_warning(f'Interpreting {len(empty_annotations)} detections without annotations as absences (0)')
        if print_annotations:
            print_warning(empty_annotations.to_string())
        raw_annotations = pd.concat([raw_annotations, empty_annotations[['species_predicted', 'label_truth', 'confidence', 'file']]], ignore_index=True)
        raw_annotations['label_truth'] = raw_annotations['label_truth'].fillna(0)

    raw_annotations = raw_annotations.sort_values(by=['species_predicted'])

    # Warn if missing annotations for any species in species_predicted
    evaluated_species = sorted(evaluated_detections['species_predicted'].astype(str).unique())
    annotated_species = sorted(raw_annotations['species_predicted'].astype(str).unique())
    missing_species = [species for species in evaluated_species if species not in annotated_species]
    if (len(missing_species) > 0):
        print_warning(f'Missing annotations for the following species:\n{missing_species}')

    # Warn for unique non-species labels
    species_classes = labels.get_species_classes()
    unique_labels = sorted(raw_annotations['label_truth'].astype(str).unique()) # Get a sorted list of unique species names
    unique_labels = [label for label in unique_labels if label not in species_classes]
    print(f'{len(unique_labels)} unique non-species labels annotated: {unique_labels}')

    if print_annotations:
        print(raw_annotations.to_string())
    
    raw_annotations.to_csv(annotations_filepath, index=False)
    print_success(f'Saved annotations to {annotations_filepath}')

    return raw_annotations
