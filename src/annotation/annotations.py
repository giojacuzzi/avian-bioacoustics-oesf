##
# Validate evaluation data and evaluate classifier performance for each species class
# Requires annotations from Google Drive. Ensure these are downloaded locally before running.

# Load required packages -------------------------------------------------
import os                       # File navigation
import pandas as pd             # Data manipulation
from utils import files, labels
from utils.log import *
import re
import sys

# Load and validate all annotations from the given directories
# Directories must contain both .txt Raven Pro selection tables and their corresponding .wav detection files
# Also performs validation
#
# Returns a dataframe of raw annotations from all Raven Pro selection tables under the given directory
def get_raw_annotations(dirs=[], overwrite=False, print_annotations=False):

    annotations_filepath = 'data/annotations/processed/annotations_processed.csv'

    # Return previously processed annotations unless overwrite requested
    if os.path.exists(annotations_filepath) and not overwrite:
        print(f'Retrieving annotations from {annotations_filepath}')
        return pd.read_csv('data/annotations/processed/annotations_processed.csv', index_col=None)
    
    if (overwrite and len(dirs) == 0) or (not overwrite and not os.path.exists(annotations_filepath) and len(dirs) == 0):
        print_error('At least one directory must be provided to write annotations to file')
        return

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
    file_metadata = list(map(lambda f: files.parse_metadata_from_annotation_file(f), evaluated_detection_files))
    species, confidences, serialnos, dates, times = zip(*file_metadata)

    evaluated_detections = pd.DataFrame({
        'file': evaluated_detection_files,
        'label_predicted': species,
        'confidence': confidences,
        'date': dates,
        'time': times,
        'serialno': serialnos
    })
    evaluated_detections['label_predicted'] = evaluated_detections['label_predicted'].str.lower()

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

        table = files.load_raven_selection_table(table_file)
        if table.empty:
            print('There was a problem with loading the selection tables. Skipping...')
            continue

        # Parse the species names and confidence scores from 'Begin File' column
        file_metadata = list(map(lambda f: files.parse_metadata_from_annotation_file(f), table['begin file']))
        species, confidences, serialnos, dates, times = zip(*file_metadata)

        table.rename(columns={'species': 'label_truth'}, inplace=True) # Rename 'species' to 'label_truth'
        table.insert(0, 'label_predicted', species) # Add column for species predicted by the classifier
        table.insert(2, 'confidence', confidences) # Add column for confidence values
        # table['serialno'] = serialnos

        # Clean up species names
        table['label_predicted'] = table['label_predicted'].str.lower()
        table['label_predicted'] = table['label_predicted'].str.replace('_', "'")
        table['label_truth'] = table['label_truth'].str.replace('_', "'")

        table.rename(columns={'begin file': 'file'}, inplace=True) # Rename 'begin file' to 'file'

        if (table['label_truth'] == 'nan').any() or table['label_truth'].isnull().any():
            print_warning(f'{os.path.basename(table_file)} contains annotations with missing labels')

        raw_annotations = pd.concat([raw_annotations, table], ignore_index=True) # Store the table

    # Warn if any annotations extend beyond the length of a single detection file
    for i, row in raw_annotations.iterrows():
        if (row['file offset (s)'] + row['delta time (s)']) > 3.0:
            print_warning(f'Annotation extends beyond detection length: {row["file"]}')

    # print(raw_annotations)

    # Include any evaluated detections for the species without annotations (which should all be FP)
    empty_annotations = evaluated_detections.merge(raw_annotations.drop(columns='label_predicted'), on=['file', 'confidence'], how='outer', indicator=True)
    empty_annotations = empty_annotations[empty_annotations['_merge'] != 'both'] # Filter out the rows where the indicator value is 'both'
    empty_annotations = empty_annotations.drop(columns='_merge')

    # Print and save empty annotations, if desired
    # print_success(empty_annotations['file'].to_string())
    # empty_annotations['file'].to_csv('empty_annotations.txt', index=False, header=False)

    if not empty_annotations.empty:
        print_warning(f'Interpreting {len(empty_annotations)} detections without annotations as species absences')
        if print_annotations:
            print_warning(empty_annotations.to_string())
        raw_annotations = pd.concat([raw_annotations, empty_annotations[['label_predicted', 'label_truth', 'confidence', 'file']]], ignore_index=True)
        raw_annotations['label_truth'] = raw_annotations['label_truth'].fillna(0)

    raw_annotations = raw_annotations.sort_values(by=['label_predicted'])

    # More clean up of typos
    print('Cleaning annotation labels...')
    raw_annotations['label_truth'] = raw_annotations['label_truth'].astype(str).apply(labels.clean_label)

    # Get metadata for each annotation
    annotations_metadata = list(map(lambda f: files.parse_metadata_from_annotation_file(f), raw_annotations['file']))
    species, confidences, serialnos, dates, times = zip(*annotations_metadata)
    raw_annotations['date'] = dates
    raw_annotations['time'] = times
    raw_annotations['serialno'] = serialnos

    # Warn if missing annotations for any species in label_predicted
    evaluated_species = sorted(evaluated_detections['label_predicted'].astype(str).unique())
    annotated_species = sorted(raw_annotations['label_predicted'].astype(str).unique())
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

# Collate raw annotations for the specified species classes, such that all annotation labels for
# each unique detection (file) are simplified to a single annotation label if the species was
# present (1), unknown (?), or not (0).
#
# Returns a dataframe of collated annotations
def collate_annotations_as_detections(raw_annotations, species_to_collate, only_annotated=False, print_detections=False):

    collated_detection_labels = pd.DataFrame()

    for i, species in enumerate(species_to_collate):

        print(f'Collating annotation data for "{species}"...')

        # Get all annotations for the species
        if True:
            # For now, only include detections (TP and FP).
            detection_labels = raw_annotations[raw_annotations['label_predicted'] == species]
        else:
            # TODO: Include not only detections (TP and FP), but also non-detections (FN).
            detection_labels = raw_annotations[(raw_annotations['label_predicted'] == species) | (raw_annotations['label_truth'] == species_to_evaluate)]
            # NOTE: The resulting confidence scores are incorrect where label_predicted != species_name
            detection_labels.loc[detection_labels['label_predicted'] != species, 'confidence'] = np.nan # Set confidence to NaN for the identified rows
            # TODO: Instead of getting confidence score from the detection file for TP and FP, get ALL confidence scores from the raw data (TP, FP, and FN). Use this to overwrite 'confidence'?
        detection_labels = detection_labels.sort_values(by='file')

        # print('All annotations:')
        # print(detection_labels)

        if detection_labels.empty:
            print_warning(f'No samples for {species}. Skipping...')
            continue

        if detection_labels['label_truth'].isna().any():
            print_warning('Assuming missing (NaN) label_truth values due to lack of annotation selections are false positives (0)...')
            if only_annotated:
                detection_labels.dropna(subset=['label_truth'], inplace=True)
                if detection_labels.empty:
                    print_warning(f'No remaining detections to evaluate for {species}. Skipping...')
                    continue
            else:
                detection_labels['label_truth'] = detection_labels['label_truth'].fillna(0)

        # TODO: When including non-detections (FN), do we need to pull confidence scores here instead? For the NaN values?

        # print('All annotations and samples:')
        # print(detection_labels)

        # For each unique detection (file), determine if the species was present (1), unknown (?), or not (0) among all annotation labels
        # Simplify the dataframe such that there is only one row per detection.
        def compute_label_presence(group, label):
            if label in group['label_truth'].values:
                return label
            elif 'unknown' in group['label_truth'].values:
                return 'unknown'
            else:
                return '0'

        # Group by the 'file' column and apply the function to compute label_presence,
        # then merge the new column back to the original DataFrame
        detection_labels = pd.merge(
            detection_labels.drop(columns=[col for col in ['label_truth', 'file offset (s)', 'delta time (s)'] if col in detection_labels.columns]),# drop selection-specific fields that no longer apply
            detection_labels.groupby('file').apply(compute_label_presence, include_groups=False, label=species).reset_index(name='label_truth'), # label_truth here will replace the previous label_truth
            on='file'
        )
        # Drop duplicate rows based on the 'file' column
        detection_labels = detection_labels.drop_duplicates(subset='file')

        if print_detections:
            print(detection_labels.to_string())

        collated_detection_labels = pd.concat([collated_detection_labels, detection_labels], ignore_index=True) # Store the detections
    
    return collated_detection_labels


# Collate raw detections from MacCaulay reference data for the specified species classes, such
# that each file is assigned a single maximum confidence score.
#
# Returns a dataframe of collated macaulay reference examples
def collate_macaulay_references(species_to_collate, print_detections=False):

    collated_detection_labels = pd.DataFrame()

    species_list = pd.read_csv(files.species_list_filepath, index_col=None, usecols=['common_name', 'id'])
    species_list['common_name'] = species_list['common_name'].str.lower()

    for i, species in enumerate(species_to_collate):

        print(f'Collating MacCaulay reference data for "{species}"...')

        # Convert species common name to ID
        species_id = species_list.loc[species_list['common_name'] == species, 'id']
        if species_id.empty:
            print_error(f'Unable to find species ID for {species}')
            continue
        else:
            species_id = species_id.iloc[0]

        # Get the list of MacCaulay files (detections)
        dir_detections = '/Users/giojacuzzi/Desktop/Macaulay/detections'
        evaluated_detection_filepaths = []
        evaluated_detection_filepaths.extend(files.find_files(dir_detections, prefix=species_id, suffix='.csv'))
        evaluated_detection_filepaths = sorted(evaluated_detection_filepaths)
        
        if len(evaluated_detection_filepaths) == 0:
            print_error(f'Unable to find references for {species}')
            continue

        # For each file, obtain the maximum detection confidence for the species 
        for file in evaluated_detection_filepaths:
            file_detections = pd.read_csv(file, index_col=None, usecols=['common_name', 'confidence'])
            file_detections['common_name'] = file_detections['common_name'].str.lower()
            file_detections = file_detections.loc[file_detections['common_name'] == species, ]
            max_confidence = file_detections['confidence'].max()

            detection_labels = pd.DataFrame({
                'label_predicted': [species],
                'confidence':        [max_confidence],
                'file':              [file],
                'label_truth':       [species]
            })
            detection_labels['confidence'] = detection_labels['confidence'].fillna(0.0)

            collated_detection_labels = pd.concat([collated_detection_labels, detection_labels], ignore_index=True) # Store the detections

    if print_detections:
        print(collated_detection_labels)

    return collated_detection_labels
