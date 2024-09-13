# Compare performance on an evaluation dataset (e.g. validation, test) between pre-trained and custom classifier
# Takes a .csv listing all validation samples as input, and assumes a directory structure of .../POSITIVE_CLASS/sample.wav
# NOTE: Must be executed from the top directory of the repo
# NOTE: Ensure you have run pretest_consolidate_test_annotations.py prior to running this!

# CHANGE ME ##############################################################################
overwrite = False
evaluation_dataset = 'test' # 'validation' or 'test'
custom_model_stub  = None # e.g. 'custom_S1_N100_A0_U0_I0' or None to only evaluate pre-trained model
##########################################################################################

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
import time

if evaluation_dataset != 'validation' and evaluation_dataset != 'test':
    print_error('Invalid evaluation dataset')
    sys.exit()
if custom_model_stub == None and evaluation_dataset == 'validation':
    print_error('Evaluating with a validation dataset requires a custom model specified')
    sys.exit()

# Load class labels
class_labels_csv_path = os.path.abspath(f'data/class_labels.csv')
class_labels = pd.read_csv(class_labels_csv_path)
if evaluation_dataset == 'validation':
    class_labels = class_labels[class_labels['train'] == 1] # only evaluate labels that were trained on
preexisting_labels_to_evaluate = list(class_labels[class_labels['novel'] == 0]['label_birdnet'])
novel_labels_to_evaluate = list(class_labels[class_labels['novel'] == 1]['label_birdnet'])
labels_to_evaluate = preexisting_labels_to_evaluate + novel_labels_to_evaluate
# print(preexisting_labels_to_evaluate)
# input()

plot = True
# DEBUG #################################################################
debug = False
debug_threshold = 1.0
if debug:
    debug_threshold = 0.0
    debug_label = "Dendragapus fuliginosus_Sooty Grouse"
    preexisting_labels_to_evaluate = [debug_label]
    labels_to_evaluate = [debug_label]
    novel_labels_to_evaluate = []
    plot = True
# DEBUG #################################################################

# TODO: Also support labels that the classifier looked for but did not find

# Output config
sort_by      = 'confidence' # Column to sort dataframe by
ascending    = False        # Column sort direction
save_to_file = True        # Save output to a file
if evaluation_dataset == 'validation' and custom_model_stub != None:
    out_dir = f'data/validation/custom/{custom_model_stub}' # Output directory (e.g. '/Users/giojacuzzi/Downloads'), if saving output to file
    custom_model_dir_path = f'data/models/custom/{custom_model_stub}'
elif evaluation_dataset == 'test':
    if custom_model_stub != None:
        out_dir = f'data/test/{custom_model_stub}'
    else:
        out_dir = 'data/test/pre-trained'
    custom_model_dir_path = f'data/models/custom/{custom_model_stub}'

# Analyzer config
min_confidence = 0.0   # Minimum confidence score to retain a detection (only used if apply_sigmoid is True)
apply_sigmoid  = True # Sigmoid transformation or raw logit score
num_separation = 1     # Number of sounds to separate for analysis. Leave as 1 for original file alone.
cleanup        = True  # Keep or remove any temporary files created through analysis
n_processes = 1

pretrained_analyzer_filepath = None # 'None' will default to the pretrained model
pretrained_labels_filepath   = 'src/classification/species_list/species_list_OESF.txt'
custom_analyzer_filepath     = f'{custom_model_dir_path}/{custom_model_stub}.tflite'
custom_labels_filepath       = f'{custom_model_dir_path}/{custom_model_stub}_Labels.txt'
training_data_path           = 'data/training'

def remove_extension(f):
    return os.path.splitext(f)[0]

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Get the evaluation data as a dataframe with columns:
    # 'path' - full path to the audio file
    # 'file' - the basename of the audio file

    if evaluation_dataset == 'validation':
        development_data = pd.read_csv(f'{custom_model_dir_path}/combined_files.csv')
        evaluation_data = development_data[development_data['dataset'] == 'validation']
        evaluation_data.loc[:, 'file'] = evaluation_data['file'].apply(remove_extension)

    elif evaluation_dataset == 'test':
        evaluation_data = pd.read_csv('data/test/test_data_annotations.csv')
        # print('Discard evaluation data for invalid class labels...')
        # print(f"before # {len(evaluation_data)}")
        # DEBUG
        evaluation_data = evaluation_data[evaluation_data['target'].isin(set(class_labels['label_birdnet']))]
        # print(labels_to_evaluate)
        # print(f"after # {len(evaluation_data)}")
        # input()
        evaluation_data['labels'] = evaluation_data['labels'].fillna('')
        # print(evaluation_data)
        # input()

    out_dir_pretrained = out_dir + '/pre-trained'
    out_dir_custom     = out_dir + '/custom'
    if custom_model_stub == None:
        models = [out_dir_pretrained]
    else:
        models = [out_dir_pretrained, out_dir_custom]

    in_evaluation_audio_filepaths = evaluation_data['path']

    print(f'Found {len(labels_to_evaluate)} labels to evaluate with {len(set(in_evaluation_audio_filepaths))} evaluation files.')
    print(sorted(labels_to_evaluate))
    # input()

    # Normalize file paths to support both mac and windows
    in_evaluation_audio_filepaths = np.vectorize(os.path.normpath)(in_evaluation_audio_filepaths)
    out_dir = os.path.normpath(out_dir)

    if overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if os.path.exists(out_dir):
        print_warning('Data already processed, loading previous results...')
    else:
        # Process evaluation examples with both classifiers -----------------------------------------------------------------------------

        def get_second_to_last_directory_of_path(path):
            parts = path.split(os.sep)
            parts = [part for part in parts if part]
            if len(parts) >= 2:
                return os.sep + os.path.join(*parts[:-2])
            else:
                return None  # Handle cases where the path doesn't contain enough components

        in_rootdirs = np.vectorize(get_second_to_last_directory_of_path)(in_evaluation_audio_filepaths)

        if custom_model_stub != None:
            # Custom classifier
            print(f"Processing evaluation set with classifier {custom_analyzer_filepath}")
            process_files.process_files_parallel(
                in_files          = in_evaluation_audio_filepaths,
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
        print(f"Processing evaluation set with classifier {pretrained_analyzer_filepath}")
        process_files.process_files_parallel(
            in_files          = in_evaluation_audio_filepaths,
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
    for model in models:
        print(f'BEGIN MODEL EVALUATION {model} ================================================================================================')

        if model == out_dir_pretrained:
            model_labels_to_evaluate = [label.split('_')[1].lower() for label in preexisting_labels_to_evaluate]
        elif model == out_dir_custom:
            model_labels_to_evaluate = [label.split('_')[1].lower() for label in labels_to_evaluate]

        print(f'Evaluating labels: {model_labels_to_evaluate}')
        # input()

        # Load analyzer detection scores for each evaluation file example
        print('Loading analyzer detection scores for evaluation examples...')
        score_files = []
        score_files.extend(files.find_files(model, '.csv')) 
        predictions = pd.DataFrame()
        i = 0
        for file in score_files:
            if i % 100 == 0:
                print(f"{round(i/len(score_files) * 100, 2)}%")
            score = pd.read_csv(file)
            score.drop(columns=['start_date'], inplace=True)
            score['file_audio'] = os.path.basename(file)
            predictions = pd.concat([predictions, score], ignore_index=True)
            i += 1
        predictions['file_audio'] = predictions['file_audio'].apply(remove_extension)
        predictions.rename(columns={'file_audio': 'file'}, inplace=True)
        predictions.rename(columns={'common_name': 'label_predicted'}, inplace=True)
        predictions['label_predicted'] = predictions['label_predicted'].str.lower()

        if evaluation_dataset == 'validation':
            # Discard scores for files used in training
            predictions = predictions[predictions['file'].isin(set(evaluation_data['file']))]
        
        # Discard prediction scores for labels not under evaluation
        print('Discard prediction scores for labels not under evaluation')
        # print(f"before # {len(predictions)}")
        # input()
        # DEBUG ################################################################################################
        predictions = predictions[predictions['label_predicted'].isin(set(model_labels_to_evaluate))]
        predictions['label_truth'] = ''
        # print(f"after # {len(predictions)}")
        # input()
        print(predictions)
        # input()

        print('Loading corresponding annotations...')
        if evaluation_dataset == 'validation':
            # Load annotation labels for the training files as dataframe with columns:
            # 'file' - the basename of the audio file
            # 'labels' - true labels, separated by ', ' token
            annotations = pd.read_csv(f'{training_data_path}/training_data_annotations.csv')
    
            # Discard annotations for files that were used in training
            print('Discarding annotations for files used in training...')
            annotations = annotations[annotations['file'].isin(set(evaluation_data['file']))]
        
        elif evaluation_dataset == 'test':
            annotations = evaluation_data
            annotations['file'] = annotations['file'].apply(remove_extension)
            print(annotations)
            # input()
        
        # Discard prediction scores for files not in evaluation dataset
        # DEBUG DEBUG DEBUG ################################################################
        print('Discard prediction scores for files not in evaluation dataset')
        # print(f"before # {len(predictions)}")
        # input()
        # predictions = predictions[predictions['file'].isin(set(evaluation_data['file']))]
        # print(f"after # {len(predictions)}")
        # input()
        # DEBUG DEBUG DEBUG ################################################################

        # At this point, "predictions" contains the model confidence score for each evaluation example for each predicted label,
        # and "annotations" contains the all annotations (i.e. true labels) for each evaluation example.

        # Next, for each label, collate annotation data into simple presence ('label_predicted') or absence ('0') truth per label prediction per file, then add to 'predictions'
        print('Collating annotations for each label...')
        count = 0
        annotations_files_set = set(annotations['file'])
        for i, row in predictions.iterrows():
            if count % 1000 == 0:
                print_success(f"{round(count/len(predictions) * 100, 2)}%")
            count += 1

            conf = row['confidence']

            # DEBUG
            if row['file'] not in annotations_files_set: # TODO: AND / INSTEAD the original target for this prediction is not in class_labels...
                # print_warning('Setting invalid file to unknown')
                predictions.at[i, 'label_truth'] = 'unknown'
                continue


            # Does the file truly contain the label?
            present = False
            unknown = False
            file_annotations = annotations[annotations['file'] == row['file']]
            # if DEBUG:
            #     print('file_annotations:')
            #     print(file_annotations)
            true_labels = []
            if len(file_annotations) == 0:
                # print_warning(f"No annotations for file {row['file']}!")
                predictions.at[i, 'label_truth'] = '0' # NOTE: Unannotated files (e.g. Background files) are intepreted as having no labels
                continue
            
            true_labels = str(file_annotations['labels'].iloc[0]).split(', ')
            if debug:
                print(f'true labels before {true_labels}')
            if len(true_labels) > 0:
                # input(f'found {len(true_labels)} labels')
                # print(f'before: {true_labels}')
                # Convert birdnet labels to simple labels
                simple_labels = []
                for label in true_labels:
                    if label not in ['unknown', 'not_target']:
                        split = label.split('_')
                        if len(split) > 1:
                            label = split[1].lower()
                    simple_labels.append(label)
                true_labels = set(simple_labels)
                # print(f'after: {true_labels}')
            
            if conf > debug_threshold and debug: # DEBUG
                print(f"row[file] {row['file']}")
                print(f'true_labels {true_labels}')

            present = row['label_predicted'] in true_labels

            if present:
                predictions.at[i, 'label_truth'] = row['label_predicted']
                # if conf > debug_threshold and debug: # DEBUG
                #     print('present')
                
            else:
                predictions.at[i, 'label_truth'] = '0'

                # if conf > debug_threshold and debug: # DEBUG
                #     print('not present')

                if 'unknown' in true_labels:
                    # print_warning('Skipping unknown example...')
                    predictions.at[i, 'label_truth'] = 'unknown'
                    # input()
                    # if conf > debug_threshold and debug: # DEBUG
                    #     print('unknown')
                elif 'not_target' in true_labels:
                    # print_warning('found a not_target')
                    # input()

                    # if conf > debug_threshold and debug: # DEBUG
                    #     print(f'not_target file_annotations {file_annotations}')

                    for j, a in file_annotations.iterrows():
                        target = a['target']
                        # print('a')
                        # print(a)
                        # input()
                        if len(target.split('_')) > 1:
                            target = a['target'].split('_')[1].lower()
                        # print(f'target {target}')
                        # input()
                        if target != row['label_predicted']:
                            # print_warning(f"Skipping unknown 'not_target' example for another label ({target})...")
                            predictions.at[i, 'label_truth'] = 'unknown'
                            # input()
                            # if conf > debug_threshold and debug: # DEBUG
                            #     print('not_target unknown')
                            break
            
            # if DEBUG:
            #     print(f'Truth: {true_labels}')
            #     print(f"Result ({row['label_predicted']}): {predictions.at[i, 'label_truth']}")
            # input()
        
        # input()

        # Interpret missing labels as absences
        if predictions['label_truth'].isna().sum() > 0:
            print(f"Intepreting {predictions['label_truth'].isna().sum()} missing labels as absences...")
            predictions['label_truth'] = predictions['label_truth'].fillna(0)

        # Drop unknown labels
        if len(predictions[predictions['label_truth'] == 'unknown']) > 0:
            print(f"Dropping {len(predictions[predictions['label_truth'] == 'unknown'])} unknown labels...")
            predictions = predictions[predictions['label_truth'] != 'unknown']

        # DEBUG
        # Save predictions to file
        print(predictions)
        print(len(predictions))

        print('Filtering...')
        # Filter out rows where 'label_truth' is 0
        pred_copy = predictions
        # df = pred_copy[pred_copy['label_truth'] != 0]
        # df = df[df['label_truth'] != '0']
        # df = pred_copy[pred_copy['label_truth'] == 'western tanager']
        # df = df[df['label_predicted'] != 'western tanager']
        # # Group by 'file' and concatenate unique 'label_truth' values
        # df = df.groupby('file').agg({'label_truth': lambda x: ', '.join(x.unique())}).reset_index()
        # # Rename the 'label_truth' column to 'labels'
        # df = df.rename(columns={'label_truth': 'labels'})
        df = predictions
        df = df[df['confidence'] > debug_threshold]
        df['serialno'] = df['file'].str.extract(r'(SMA\d{5})')
        df['month'] = df['file'].str.extract(r'(_2020\d{2})')
        # df = df.sort_values(by=['serialno', 'month', 'file']).reset_index(drop=True)
        df = df.sort_values(by=['confidence']).reset_index(drop=True)
        df.to_csv('/Users/giojacuzzi/Downloads/test_labels_revised.csv')
        print(f'Down to {len(df)} files')
        # input()
        # DEBUG

        # Use 'predictions' to evaluate performance metrics for each label
        model_performance_metrics = pd.DataFrame() # Container for performance metrics of all labels
        for label_under_evaluation in model_labels_to_evaluate:
            print(f"Calculating performance metrics for '{label_under_evaluation}'...")

            # Get all the predictions and their associated confidence scores for this label
            detection_labels = predictions[predictions['label_predicted'] == label_under_evaluation]

            species_performance_metrics = evaluate_species_performance(detection_labels=detection_labels, species=label_under_evaluation, plot=plot, title_label=model, save_to_dir=f'/Users/giojacuzzi/Downloads/perf/{model}')
            model_performance_metrics = pd.concat([model_performance_metrics, species_performance_metrics], ignore_index=True)

        model_performance_metrics['model'] = model
        print(f'PERFORMANCE METRICS FOR {model}')
        print(model_performance_metrics.to_string())

        if model == out_dir_pretrained:
            model_performance_metrics.to_csv(f"{out_dir}/metrics_pre-trained.csv")
        elif model == out_dir_custom:
            model_performance_metrics.to_csv(f"{out_dir}/metrics_custom.csv")

        performance_metrics = pd.concat([performance_metrics, model_performance_metrics], ignore_index=True)

        if model == out_dir_pretrained:
            collated_detection_labels_pretrained = predictions
        elif model == out_dir_custom:
            collated_detection_labels_custom = predictions

    print('FINAL RESULTS ================================================================================================')
    # performance_metrics.sort_values(by=['label', 'model'], inplace=True)
    performance_metrics.sort_values(by=['AUC-PR'], inplace=True)
    print(performance_metrics.to_string())
    if plot:
        plt.show()
        input('plot?')

    # Calculate metric deltas between custom and pre-trained
    if len(models) > 1:
        print('Deltas between custom and pre-trained:')
        metrics_custom = performance_metrics[performance_metrics['model'] == out_dir_custom][['label', 'AUC-PR', 'AUC-ROC', 'f1_max', 'p_max_r']].rename(columns={'AUC-PR': 'AUC-PR_custom', 'AUC-ROC': 'AUC-ROC_custom', 'f1_max': 'f1_max_custom', 'p_max_r': 'p_max_r_custom'})
        metrics_pre_trained = performance_metrics[performance_metrics['model'] == out_dir_pretrained][['label', 'AUC-PR', 'AUC-ROC', 'f1_max', 'p_max_r']].rename(columns={'AUC-PR': 'AUC-PR_pre_trained', 'AUC-ROC': 'AUC-ROC_pre_trained', 'f1_max': 'f1_max_pre_trained', 'p_max_r': 'p_max_r_pre_trained'})
        delta_metrics = pd.merge(metrics_custom, metrics_pre_trained, on='label')
        delta_metrics['AUC-PR_diff'] = delta_metrics['AUC-PR_custom'] - delta_metrics['AUC-PR_pre_trained']
        delta_metrics['AUC-ROC_diff'] = delta_metrics['AUC-ROC_custom'] - delta_metrics['AUC-ROC_pre_trained']
        delta_metrics['f1_max_diff'] = delta_metrics['f1_max_custom'] - delta_metrics['f1_max_pre_trained']
        delta_metrics['p_max_r_diff'] = delta_metrics['p_max_r_custom'] - delta_metrics['p_max_r_pre_trained']
        delta_metrics = delta_metrics.sort_index(axis=1)
        col_order = ['label'] + [col for col in delta_metrics.columns if col != 'label']
        delta_metrics = delta_metrics[col_order]
        # print(delta_metrics)

        # Calculate macro-averaged metrics for each model
        mean_values = delta_metrics.drop(columns='label').mean()
        mean_row = pd.Series(['MEAN'] + mean_values.tolist(), index=delta_metrics.columns)
        delta_metrics = pd.concat([delta_metrics, pd.DataFrame([mean_row])], ignore_index=True)

        print(delta_metrics)
        delta_metrics.to_csv(f"{out_dir}/metrics_summary.csv")

    # Site-level performance ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #     - Number of sites detected / number of sites truly present
    #     - Number of sites not detected / number of sites truly present
    #     - Number of sites detected / number of sites truly absent

    # TODO

    sys.exit()

        # Get serialno, date, and time
    predictions[['serialno', 'date', 'time']] = predictions['file'].apply(lambda f: pd.Series(parse_metadata_from_detection_audio_filename(f)))
    predictions['date'] = pd.to_datetime(predictions['date'], format='%Y%m%d')
    print(predictions)
    print(predictions['label_truth'].value_counts())        
    print(f"{len(predictions['file'].unique())} unique files in predictions")
    print(f"{len(predictions['label_truth'].unique())} unique labels in predictions")
    print(f'{len(predictions)} predictions in total')

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
