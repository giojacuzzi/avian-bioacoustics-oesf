# Compare performance on validation dataset between pre-traned and custom classifier
# Takes a .csv listing all validation samples as input, and assumes a directory structure of .../POSITIVE_CLASS/sample.wav
# NOTE: Must be executed from the top directory of the repo

from classification import process_files
from utils.log import *
import os
import numpy as np
import pandas as pd
from utils import files
from utils import labels
from annotation.annotations import *
from classification.performance import *
import matplotlib.pyplot as plt
import sys

# Input config
validation_samples_filepath = 'data/models/Custom/Custom_Classifier_ValidationSamples.csv'
classes_to_evaluate = ["red-breasted nuthatch"]

# Output config
sort_by      = 'confidence' # Column to sort dataframe by
ascending    = False        # Column sort direction
save_to_file = False        # Save output to a file
out_dir      = 'data/validation/Custom' # Output directory (e.g. '/Users/giojacuzzi/Downloads'), if saving output to file

# Analyzer config
min_confidence = 0.0   # Minimum confidence score to retain a detection (only used if apply_sigmoid is True)
apply_sigmoid  = True # Sigmoid transformation or raw logit score
num_separation = 1     # Number of sounds to separate for analysis. Leave as 1 for original file alone.
cleanup        = True  # Keep or remove any temporary files created through analysis
n_processes = 8

pretrained_analyzer_filepath = None
pretrained_labels_filepath   = 'src/classification/species_list/species_list_OESF.txt'
custom_analyzer_filepath     = 'data/models/Custom/Custom_Classifier.tflite'
custom_labels_filepath       = 'data/models/Custom/Custom_Classifier_Labels.txt'
training_data_selections_dir = 'data/training/Custom/selections'

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    out_dir_pretrained = out_dir + '/pre-trained'
    out_dir_custom     = out_dir + '/custom'

    in_filepaths = np.genfromtxt(validation_samples_filepath, delimiter=',', dtype=str)

    # Normalize file paths to support both mac and windows
    in_filepaths = np.vectorize(os.path.normpath)(in_filepaths)
    out_dir      = os.path.normpath(out_dir)

    if os.path.exists(out_dir):
        print_warning('Data already processed, loading previous results...')
    else:
        # Process validation examples with both classifiers -----------------------------------------------------------------------------

        def get_second_to_last_directory(path):
            parts = path.split(os.sep)
            parts = [part for part in parts if part]
            if len(parts) >= 2:
                return os.sep + os.path.join(*parts[:-2])
            else:
                return None  # Handle cases where the path doesn't contain enough components

        in_rootdirs = np.vectorize(get_second_to_last_directory)(in_filepaths)

        # Pre-trained classifier
        process_files.process_files_parallel(
            in_files          = in_filepaths,
            out_dir           = out_dir_pretrained,
            root_dirs         = in_rootdirs,
            in_filetype       = '.wav',
            analyzer_filepath = pretrained_analyzer_filepath,
            labels_filepath   = pretrained_labels_filepath,
            n_processes       = n_processes,
            min_confidence    = min_confidence,
            apply_sigmoid     = apply_sigmoid,
            num_separation    = num_separation,
            cleanup           = cleanup,
            sort_by           = sort_by,
            ascending         = ascending
        )

        # Pre-trained classifier
        process_files.process_files_parallel(
            in_files          = in_filepaths,
            out_dir           = out_dir_custom,
            root_dirs         = in_rootdirs,
            in_filetype       = '.wav',
            analyzer_filepath = custom_analyzer_filepath,
            labels_filepath   = custom_labels_filepath,
            n_processes       = n_processes,
            min_confidence    = min_confidence,
            apply_sigmoid     = apply_sigmoid,
            num_separation    = num_separation,
            cleanup           = cleanup,
            sort_by           = sort_by,
            ascending         = ascending
        )

        print(f'Finished analyzing all files!')
    
    # TODO: Load results per classifier and calculate performance stats ---------------------------------------------------------------
    for model in [out_dir_pretrained, out_dir_custom]:
        print(f'Evaluating model {model}...')
        todo = model

        def remove_extension(f):
            return os.path.splitext(f)[0]

        # Load analyzer detection scores
        print('Loading analyzer detection scores for validation examples...')
        score_files = []
        score_files.extend(files.find_files(todo, '.csv')) 
        scores = pd.DataFrame()
        for file in score_files:
            score = pd.read_csv(file)
            score.drop(columns=['start_date'], inplace=True)
            score['file_audio'] = os.path.basename(file)
            scores = pd.concat([scores, score], ignore_index=True)
        scores['file_audio'] = scores['file_audio'].apply(remove_extension)
        scores.rename(columns={'common_name': 'label_predicted'}, inplace=True)
        scores['label_predicted'] = scores['label_predicted'].str.lower()
        # print(scores.to_string())

        # Load selection table files
        print('Loading corresponding selection table files...')
        selection_table_files = []
        selection_table_files.extend(files.find_files(training_data_selections_dir, '.txt'))
        annotations = pd.DataFrame()
        for file in selection_table_files:
            selections = files.load_raven_selection_table(file, cols_needed = ['class', 'file_audio']) # true labels
            annotations = pd.concat([annotations, selections], ignore_index=True)
        annotations['file_audio'] = annotations['file_audio'].apply(remove_extension)
        annotations.rename(columns={'class': 'label_truth'}, inplace=True)
        annotations['label_truth'] = annotations['label_truth'].str.lower()
        # print(annotations.to_string())

        # Merge, discarding annotations for non-validation files that were used in training
        print('Merging...')
        detections = pd.merge(scores, annotations, on=['file_audio'], how='left')
        detections.sort_values(by='file_audio', inplace=True)
        detections.rename(columns={'file_audio': 'file'}, inplace=True)
        detections['label_truth'] = detections['label_truth'].fillna('0') # interpret missing annotations as absence
        detections['label_truth'] = detections['label_truth'].apply(labels.clean_label)

        # Collate raw annotation data into species detection labels per species
        print('Collating annotations per label...')
        collated_detection_labels = collate_annotations_as_detections(detections, classes_to_evaluate, only_annotated=False)
        # print_success(collated_detection_labels.to_string())

        # Containers for performance metrics of all labels
        performance_metrics = pd.DataFrame()

        for class_under_evaluation in classes_to_evaluate:
            print(f'Calculating performance metrics for class {class_under_evaluation}...')

            detection_labels = collated_detection_labels[collated_detection_labels['label_predicted'] == class_under_evaluation]
            species_performance_metrics = evaluate_species_performance(detection_labels, class_under_evaluation, True)
            performance_metrics = pd.concat([performance_metrics, species_performance_metrics], ignore_index=True)
        
        print(performance_metrics.to_string())

    plt.show()
