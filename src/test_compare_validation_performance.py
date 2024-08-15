# Compare performance on validation dataset between pre-traned and custom classifier
# Takes a .csv listing all validation samples as input, and assumes a directory structure of .../POSITIVE_CLASS/sample.wav
# NOTE: Must be executed from the top directory of the repo

from classification import process_files
from utils.log import *
from utils.files import *
import os
import numpy as np
import pandas as pd
from utils import files
from utils import labels
from annotation.annotations import *
from classification.performance import *
import matplotlib.pyplot as plt
import shutil
import sys

# Input config
validation_samples_filepath = 'data/models/Custom/Custom_Classifier_ValidationSamples.csv'
preexisting_labels_to_evaluate = [
    "american robin",
    "band-tailed pigeon",
    "barred owl",
    "belted kingfisher",
    "black-throated gray warbler",
    "common raven",
    "dark-eyed junco",
    "golden-crowned kinglet",
    "hairy woodpecker",
    "hammond's flycatcher",
    "hermit thrush",
    "hutton's vireo",
    "marbled murrelet",
    "northern flicker",
    "northern pygmy-owl",
    "northern saw-whet owl",
    "olive-sided flycatcher",
    "pacific wren",
    "pacific-slope flycatcher",
    "pileated woodpecker",
    "purple finch",
    "red crossbill",
    "red-breasted nuthatch",
    "ruby-crowned kinglet",
    "rufous hummingbird",
    "song sparrow",
    "sooty grouse",
    "spotted towhee",
    "swainson's thrush",
    "townsend's warbler",
    "varied thrush",
    "violet-green swallow",
    "western screech-owl",
    "western tanager",
    "western wood-pewee",
    "white-crowned sparrow",
    "wilson's warbler"
]
novel_labels_to_evaluate = [
    "abiotic_aircraft",
    "abiotic_logging",
    "abiotic_rain",
    "abiotic_vehicle",
    "abiotic_wind",
    "biotic_anuran",
    "biotic_insect"
]
labels_to_evaluate = preexisting_labels_to_evaluate + novel_labels_to_evaluate

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
n_processes = 1

pretrained_analyzer_filepath = None
pretrained_labels_filepath   = 'src/classification/species_list/species_list_OESF.txt'
custom_analyzer_filepath     = 'data/models/Custom/Custom_Classifier.tflite'
custom_labels_filepath       = 'data/models/Custom/Custom_Classifier_Labels.txt'
training_data_selections_dir = 'data/training/Custom/selections'

overwrite = False

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    out_dir_pretrained = out_dir + '/pre-trained'
    out_dir_custom     = out_dir + '/custom'

    in_filepaths = np.genfromtxt(validation_samples_filepath, delimiter=',', dtype=str)

    # Normalize file paths to support both mac and windows
    in_filepaths = np.vectorize(os.path.normpath)(in_filepaths)
    out_dir      = os.path.normpath(out_dir)

    if overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

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

        # Custom classifier
        print(f"Processing validation set with classifier {custom_analyzer_filepath}")
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

        # Pre-trained classifier
        print(f"Processing validation set with classifier {pretrained_analyzer_filepath}")
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

        print(f'Finished analyzing all files!')
    
    performance_metrics = pd.DataFrame()

    collated_detection_labels_pretrained = pd.DataFrame()
    collated_detection_labels_custom = pd.DataFrame()
    
    # Load results per classifier and calculate performance stats ---------------------------------------------------------------
    for model in [out_dir_pretrained, out_dir_custom]:
        print(f'Evaluating model {model}...')

        def remove_extension(f):
            return os.path.splitext(f)[0]

        # Load analyzer detection scores
        print('Loading analyzer detection scores for validation examples...')
        score_files = []
        score_files.extend(files.find_files(model, '.csv')) 
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
            selections = files.load_raven_selection_table(file, cols_needed = ['label', 'file_audio']) # true labels
            annotations = pd.concat([annotations, selections], ignore_index=True)
        annotations['file_audio'] = annotations['file_audio'].apply(remove_extension)
        annotations.rename(columns={'label': 'label_truth'}, inplace=True)
        annotations['label_truth'] = annotations['label_truth'].str.lower()
        # print(annotations.to_string())

        # Merge, discarding annotations for non-validation files that were used in training
        print('Merging...')
        detections = pd.merge(scores, annotations, on=['file_audio'], how='left')
        # print('Sorting...')
        # detections.sort_values(by='file_audio', inplace=True)
        detections.rename(columns={'file_audio': 'file'}, inplace=True)
        detections['label_truth'] = detections['label_truth'].fillna('0') # interpret missing annotations as absence
        detections['label_truth'] = detections['label_truth'].apply(labels.clean_label)

        # Collate raw annotation data into species detection labels per species
        print('Collating annotations per label...')
        collated_detection_labels = collate_annotations_as_detections(detections, labels_to_evaluate, only_annotated=False)
        # print(collated_detection_labels)

        # Get detection serialno, date, and time
        collated_detection_labels[['serialno', 'date', 'time']] = collated_detection_labels['file'].apply(lambda f: pd.Series(parse_metadata_from_detection_audio_filename(f)))
        # print(collated_detection_labels)

        # Containers for performance metrics of all labels
        model_performance_metrics = pd.DataFrame()

        for label_under_evaluation in labels_to_evaluate:
            print(f'Calculating performance metrics for label {label_under_evaluation}...')

            if model == out_dir_pretrained and label_under_evaluation in novel_labels_to_evaluate:
                print_warning(f"Skipping evaluation of invalid label {label_under_evaluation} for pretrained model...")
                continue

            detection_labels = collated_detection_labels[collated_detection_labels['label_predicted'] == label_under_evaluation]
            species_performance_metrics = evaluate_species_performance(detection_labels, label_under_evaluation, True, title_label=model, save_to_dir=f'/Users/giojacuzzi/Downloads/perf/{model}')
            model_performance_metrics = pd.concat([model_performance_metrics, species_performance_metrics], ignore_index=True)
        
        model_performance_metrics['model'] = model
        print(model_performance_metrics.to_string())

        performance_metrics = pd.concat([performance_metrics, model_performance_metrics], ignore_index=True)

        collated_detection_labels['date'] = pd.to_datetime(collated_detection_labels['date'], format='%Y%m%d')
        if model == out_dir_pretrained:
            collated_detection_labels_pretrained = collated_detection_labels
        elif model == out_dir_custom:
            collated_detection_labels_custom = collated_detection_labels

    print('FINAL RESULTS')
    performance_metrics.sort_values(by=['label', 'model'], inplace=True)
    print(performance_metrics.to_string())
    # plt.show()

    # Calculate metric deltas between custom and pre-trained
    print('Deltas between custom and pre-trained:')
    metrics_custom = performance_metrics[performance_metrics['model'] == 'data/validation/Custom/custom'][['label', 'AUC-PR', 'p_max_r']].rename(columns={'AUC-PR': 'AUC-PR_custom', 'p_max_r': 'p_max_r_custom'})
    metrics_pre_trained = performance_metrics[performance_metrics['model'] == 'data/validation/Custom/pre-trained'][['label', 'AUC-PR', 'p_max_r']].rename(columns={'AUC-PR': 'AUC-PR_pre_trained', 'p_max_r': 'p_max_r_pre_trained'})
    delta_metrics = pd.merge(metrics_custom, metrics_pre_trained, on='label')
    delta_metrics['AUC-PR_diff'] = delta_metrics['AUC-PR_custom'] - delta_metrics['AUC-PR_pre_trained']
    delta_metrics['p_max_r_diff'] = delta_metrics['p_max_r_custom'] - delta_metrics['p_max_r_pre_trained']
    print(delta_metrics)

    # Site-level performance ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #     - Number of sites detected / number of sites truly present
    #     - Number of sites not detected / number of sites truly present
    #     - Number of sites detected / number of sites truly absent

    # TODO

    site_deployment_metadata = get_site_deployment_metadata(2020)

    detections_custom = pd.merge(
        site_deployment_metadata, 
        collated_detection_labels_custom, 
        left_on=['SerialNo', 'SurveyDate'], 
        right_on=['serialno', 'date'],
        how='inner'
    )
    detections_pretrained = pd.merge(
        site_deployment_metadata, 
        collated_detection_labels_pretrained, 
        left_on=['SerialNo', 'SurveyDate'], 
        right_on=['serialno', 'date'],
        how='inner'
    )

    all_sites = detections_custom["StationName"].unique()
    ntotal_sites = len(all_sites)
    print(f'{ntotal_sites} unique sites evaluated')
    if (ntotal_sites != 16):
        print_warning('Missing sites')

    if True:
        site_level_perf = pd.DataFrame()
        for label in preexisting_labels_to_evaluate:
            print(f'Calculating site-level performance metrics for label {label} with custom vs pretrained classifier ...')

            pmax_th_custom = performance_metrics[(performance_metrics['label'] == label) & (performance_metrics['model'] == out_dir_custom)]['p_max_th'].iloc[0]
            species_perf_custom = get_site_level_confusion_matrix(label, detections_custom, pmax_th_custom, all_sites)
            species_perf_custom['model'] = 'custom'
            site_level_perf = pd.concat([site_level_perf, species_perf_custom], ignore_index=True)

            pmax_th_pretrained = performance_metrics[(performance_metrics['label'] == label) & (performance_metrics['model'] == out_dir_pretrained)]['p_max_th'].iloc[0]
            species_perf_pretrained = get_site_level_confusion_matrix(label, detections_pretrained, pmax_th_pretrained, all_sites)
            species_perf_pretrained['model'] = 'pretrained'
            site_level_perf = pd.concat([site_level_perf, species_perf_pretrained], ignore_index=True)

        print('Site-level performance metrics:')
        print_success(site_level_perf[(site_level_perf['present'] > 2)].sort_values(by=['label', 'model'], ascending=True).to_string())
