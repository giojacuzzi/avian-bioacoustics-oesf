# Evaluate performance of a model on an evaluation dataset (e.g. validation, test), and compare performance with a pre-trained model
# Takes a .csv listing all validation samples as input, and assumes a directory structure of .../POSITIVE_CLASS/sample.wav
# NOTE: Must be executed from the top directory of the repo
# NOTE: Ensure you have run pretest_consolidate_test_annotations.py prior to running this!

# Afterwards, plot results with pr_performance.R

# CHANGE ME ##############################################################################
overwrite = False
evaluation_dataset = 'test' # 'validation' or 'test'
custom_model_stub  = 'custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0' # e.g. 'custom_S1_N100_LR0.001_BS10_HU0_LSFalse_US0_I0' or None to only evaluate pre-trained model
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
target_labels_to_evaluate = list(class_labels[class_labels['train'] == 1]['label_birdnet'])
# all_labels = preexisting_labels_to_evaluate + novel_labels_to_evaluate
print(f'{len(preexisting_labels_to_evaluate)} preexisting labels to evaluate:')
print(preexisting_labels_to_evaluate)
# input()

plot_precision_recall = False
plot_score_distributions = False
# DEBUG #################################################################
debug = False
debug_threshold = 1.0
if debug:
    debug_threshold = 0.0
    debug_label = "Biotic_Biotic Anuran" # e.g. "Biotic_Biotic Anuran"
    preexisting_labels_to_evaluate = [debug_label]
    target_labels_to_evaluate = [debug_label]
    plot_precision_recall = True

# preexisting_labels_to_evaluate = ['Bubo virginianus_Great Horned Owl', 'Junco hyemalis_Dark-eyed Junco', 'Regulus satrapa_Golden-crowned Kinglet', 'Dendragapus fuliginosus_Sooty Grouse', 'Dryobates pubescens_Downy Woodpecker']
# novel_labels_to_evaluate = ['Abiotic_Abiotic Aircraft']
# target_labels_to_evaluate = ['Abiotic_Abiotic Aircraft', 'Junco hyemalis_Dark-eyed Junco', 'Regulus satrapa_Golden-crowned Kinglet', 'Dendragapus fuliginosus_Sooty Grouse']
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
apply_sigmoid  = False # Sigmoid transformation or raw logit score
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

    print(f'PERFORMANCE EVALUATION - vocalization level ================================================================================================')

    # Get the evaluation data as a dataframe with columns:
    # 'path' - full path to the audio file
    # 'file' - the basename of the audio file

    if evaluation_dataset == 'validation':
        development_data = pd.read_csv(f'{custom_model_dir_path}/combined_files.csv')
        evaluation_data = development_data[development_data['dataset'] == 'validation']
        evaluation_data.loc[:, 'file'] = evaluation_data['file'].apply(remove_extension)

    elif evaluation_dataset == 'test':
        evaluation_data = pd.read_csv('data/test/test_data_annotations.csv') # see pretest_consolidate_test_annotations.py
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

    # print(f'Found {len(labels_to_evaluate)} labels to evaluate with {len(set(in_evaluation_audio_filepaths))} evaluation files.')
    # print(sorted(labels_to_evaluate))
    # input()

    # Normalize file paths to support both mac and windows
    in_evaluation_audio_filepaths = np.vectorize(os.path.normpath)(in_evaluation_audio_filepaths)
    out_dir = os.path.normpath(out_dir)

    if overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if os.path.exists(out_dir):
        print_warning('Raw audio data already processed, loading previous detection results for model(s)...')
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
            print(f"Processing evaluation set with classifier {custom_analyzer_filepath}..................................................")
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
        print(f"Processing evaluation set with classifier {pretrained_analyzer_filepath}..................................................")
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

    # collated_detection_labels_pretrained = pd.DataFrame()
    # collated_detection_labels_custom = pd.DataFrame()
    
    # Load results per classifier and calculate performance stats ---------------------------------------------------------------
    for model in models:
        print(f'BEGIN MODEL EVALUATION {model} (sample level) ---------------------------------------------------------------------------')

        if model == out_dir_pretrained:
            model_labels_to_evaluate = [label.split('_')[1].lower() for label in preexisting_labels_to_evaluate]
        elif model == out_dir_custom:
            model_labels_to_evaluate = [label.split('_')[1].lower() for label in target_labels_to_evaluate]

        print(f'Evaluating labels: {model_labels_to_evaluate}')
        # input()

        # Load analyzer detection scores for each evaluation file example
        print(f'Loading {model} detection scores for evaluation examples...')
        score_files = []
        score_files.extend(files.find_files(model, '.csv', exclude_dirs=['threshold_perf'])) 
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
        print('Discarding prediction scores for labels not under evaluation...')
        l_before = predictions['label_predicted'].unique()
        # print(f"before # {len(predictions)}")
        # print(predictions['label_predicted'].unique())
        # input()
        # DEBUG ################################################################################################
        predictions = predictions[predictions['label_predicted'].isin(set(model_labels_to_evaluate))]
        predictions['label_truth'] = ''
        # print(f"after # {len(predictions)}")
        # print(predictions['label_predicted'].unique())
        # l_after = predictions['label_predicted'].unique()
        # print(f'discarded: {np.setdiff1d(l_before, l_after)}')
        # input()

        # print('after predictions:')
        # print(predictions)
        # input()

        if model == out_dir_pretrained:
            raw_predictions_pretrained = predictions
        elif model == out_dir_custom:
            raw_predictions_custom = predictions

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
            annotations = evaluation_data.copy()
            annotations['file'] = annotations['file'].apply(remove_extension)
            # print(annotations)
            # input()

        # Discard prediction scores for files not in evaluation dataset
        # DEBUG DEBUG DEBUG ################################################################
        # print('Discard prediction scores for files not in evaluation dataset')
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
                print(f"{round(count/len(predictions) * 100, 2)}%")
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
                predictions.at[i, 'label_truth'] = '0' # NOTE: Unannotated files (e.g. Background files) are intepreted as having no relevant signals (i.e. labels) present
                continue
            
            true_labels = str(file_annotations['labels'].iloc[0]).split(', ')
            # if debug:
            #     print(f'true labels before {true_labels}')
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
            
            # if conf > debug_threshold and debug: # DEBUG
            #     print(f"row[file] {row['file']}")
            #     print(f'true_labels {true_labels}')

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
        
        # print(f'DEBUG {model} predictions....')
        # print(predictions)
        # print('unique:')
        # print(predictions['label_truth'].unique())
        # input()

        # Interpret missing labels as absences
        if predictions['label_truth'].isna().sum() > 0:
            print(f"Intepreting {predictions['label_truth'].isna().sum()} predictions with missing labels as absences...")
            predictions['label_truth'] = predictions['label_truth'].fillna(0)

        # if model == out_dir_pretrained:
        #     collated_predictions_pretrained = predictions
        #     collated_predictions_pretrained.to_csv('/Users/giojacuzzi/Downloads/collated_predictions_pretrained.csv')
        # elif model == out_dir_custom:
        #     collated_predictions_custom = predictions
        #     collated_predictions_custom.to_csv('/Users/giojacuzzi/Downloads/collated_predictions_custom.csv')

        # Drop unknown labels
        if len(predictions[predictions['label_truth'] == 'unknown']) > 0:
            print(f"Dropping {len(predictions[predictions['label_truth'] == 'unknown'])} predictions with unknown labels...")
            predictions = predictions[predictions['label_truth'] != 'unknown']
        
        # print('DEBUG post unknown drop')
        # print(predictions)
        # input()

        # DEBUG
        # Save predictions to file
        # print(predictions)
        # print(len(predictions))

        # print('Filtering...')
        # Filter out rows where 'label_truth' is 0
        # pred_copy = predictions
        # df = pred_copy[pred_copy['label_truth'] != 0]
        # df = df[df['label_truth'] != '0']
        # df = pred_copy[pred_copy['label_truth'] == 'western tanager']
        # df = df[df['label_predicted'] != 'western tanager']
        # # Group by 'file' and concatenate unique 'label_truth' values
        # df = df.groupby('file').agg({'label_truth': lambda x: ', '.join(x.unique())}).reset_index()
        # # Rename the 'label_truth' column to 'labels'
        # df = df.rename(columns={'label_truth': 'labels'})
        if debug:
            df = predictions
            df = df[df['confidence'] > debug_threshold]
            df['serialno'] = df['file'].str.extract(r'(SMA\d{5})')
            df['month'] = df['file'].str.extract(r'(_2020\d{2})')
            # df = df.sort_values(by=['serialno', 'month', 'file']).reset_index(drop=True)
            df = df.sort_values(by=['confidence']).reset_index(drop=True)
            df.to_csv('/Users/giojacuzzi/Downloads/test_labels_revised.csv')
            input('Saved test labels to file! [Press return to continue]')

        # print(f'Down to {len(df)} files')
        # input()
        # DEBUG

        # Use 'predictions' to evaluate performance metrics for each label
        model_performance_metrics = pd.DataFrame() # Container for performance metrics of all labels
        for label_under_evaluation in model_labels_to_evaluate:
            print(f"Calculating performance metrics for '{label_under_evaluation}'...")

            # Get all the predictions and their associated confidence scores for this label
            detection_labels = predictions[predictions['label_predicted'] == label_under_evaluation]

            species_performance_metrics = evaluate_species_performance(detection_labels=detection_labels, species=label_under_evaluation, plot=plot_precision_recall, title_label=model, save_to_dir=f'{model}/threshold_perf')
            model_performance_metrics = pd.concat([model_performance_metrics, species_performance_metrics], ignore_index=True)

        model_performance_metrics['model'] = model
        performance_metrics = pd.concat([performance_metrics, model_performance_metrics], ignore_index=True)

        # Display results and save to file
        model_performance_metrics[model_performance_metrics.select_dtypes(include='number').columns] = model_performance_metrics.select_dtypes(include='number').round(2)
        if not debug:
            model_performance_metrics['label'] = model_performance_metrics['label'].str.title()
            model_performance_metrics.loc[model_performance_metrics['N_pos'] == 0, ['PR_AUC', 'AP', 'ROC_AUC', 'f1_max']] = np.nan
            model_performance_metrics = model_performance_metrics.sort_values(by=['PR_AUC'], ascending=False).reset_index(drop=True)
        print(f'PERFORMANCE METRICS FOR {model}')
        print(model_performance_metrics.to_string())

        if model == out_dir_pretrained:
            fp = f"{out_dir}/metrics_pre-trained.csv"
        elif model == out_dir_custom:
            fp = f"{out_dir}/metrics_custom.csv"
        model_performance_metrics.to_csv(fp, index=False)
        print_success(f'Results saved to {fp}')

        if plot_precision_recall:
            plt.show()

        # if model == out_dir_pretrained:
        #     collated_detection_labels_pretrained = predictions
        # elif model == out_dir_custom:
        #     collated_detection_labels_custom = predictions
        
        # input()

    print('FINAL RESULTS (vocalization level) ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    # performance_metrics.sort_values(by=['label', 'model'], inplace=True)
    performance_metrics = performance_metrics.drop_duplicates()
    performance_metrics.sort_values(by=['PR_AUC', 'label', 'model'], inplace=True)

    performance_metrics_out = performance_metrics
    performance_metrics_out[performance_metrics_out.select_dtypes(include='number').columns] = performance_metrics_out.select_dtypes(include='number').round(2)
    if not debug:
        performance_metrics_out['label'] = performance_metrics_out['label'].str.title()
        performance_metrics_out.loc[performance_metrics_out['N_pos'] == 0, ['PR_AUC', 'AP', 'ROC_AUC', 'f1_max']] = np.nan
        performance_metrics_out = performance_metrics_out.sort_values(by=['model', 'PR_AUC'], ascending=False).reset_index(drop=True)

    print(performance_metrics_out.to_string())
    fp = '/Users/giojacuzzi/Downloads/performance_metrics.csv'
    performance_metrics_out.to_csv(fp, index=False)
    print_success(f'Saved combined performance results to {fp}')

    # Calculate metric deltas between custom and pre-trained
    if len(models) > 1:
        # TODO: Ensure these are comparing the same shared labels?
        print('Deltas between custom and pre-trained:')
        metrics_custom = performance_metrics[performance_metrics['model'] == out_dir_custom][[
            'label', 'AP', 'PR_AUC', 'ROC_AUC', 'f1_max'
        ]].rename(columns={
            'AP': 'AP_custom', 'PR_AUC': 'PR_AUC_custom', 'ROC_AUC': 'ROC_AUC_custom', 'f1_max': 'f1_max_custom'
        })
        metrics_pre_trained = performance_metrics[performance_metrics['model'] == out_dir_pretrained][[
            'label', 'AP', 'PR_AUC', 'ROC_AUC', 'f1_max'
        ]].rename(columns={
            'AP': 'AP_pre_trained', 'PR_AUC': 'PR_AUC_pre_trained', 'ROC_AUC': 'ROC_AUC_pre_trained', 'f1_max': 'f1_max_pre_trained'
        })
        delta_metrics = pd.merge(metrics_custom, metrics_pre_trained, on='label')
        delta_metrics['AP_Δ']      = delta_metrics['AP_custom'] - delta_metrics['AP_pre_trained']
        delta_metrics['PR_AUC_Δ']  = delta_metrics['PR_AUC_custom'] - delta_metrics['PR_AUC_pre_trained']
        delta_metrics['ROC_AUC_Δ'] = delta_metrics['ROC_AUC_custom'] - delta_metrics['ROC_AUC_pre_trained']
        delta_metrics['f1_max_Δ']  = delta_metrics['f1_max_custom'] - delta_metrics['f1_max_pre_trained']
        delta_metrics = delta_metrics.sort_index(axis=1)
        col_order = ['label'] + [col for col in delta_metrics.columns if col != 'label']
        delta_metrics = delta_metrics[col_order]
        delta_metrics = delta_metrics.drop_duplicates()
        delta_metrics = delta_metrics.sort_values(by=['PR_AUC_Δ'], ascending=False).reset_index(drop=True)
        # print(delta_metrics)

        # Calculate macro-averaged metrics for each model
        mean_values = delta_metrics.drop(columns='label').mean()
        mean_row = pd.Series(['MEAN'] + mean_values.tolist(), index=delta_metrics.columns)
        delta_metrics = pd.concat([delta_metrics, pd.DataFrame([mean_row])], ignore_index=True)

        # Format results
        delta_metrics[delta_metrics.select_dtypes(include='number').columns] = delta_metrics.select_dtypes(include='number').round(2)
        delta_metrics['label'] = delta_metrics['label'].str.title()

        print(delta_metrics)
        fp = f"{out_dir}/metrics_summary.csv"
        delta_metrics.to_csv(fp, index=False)
        print_success(f'Results saved to {fp}')

    # TODO: For each label in 'predictions', perform Hartigan's dip test for multimodality with raw logit scores
    # If p_value < 0.05 we reject the null hypothesis of unimodality and conclude the data for that label is likely multimodal
    if False:
        from diptest import diptest
        for label in [label.split('_')[1].lower() for label in preexisting_labels_to_evaluate]:
            print(f'Performing dip test for {label}...')
            print('PRETRAINED')
            pretrained_scores = raw_predictions_pretrained[raw_predictions_pretrained['label_predicted'] == label]
            pretrained_dipstat, pretrained_p_value = diptest(pretrained_scores['logit'])
            print("Dip statistic:", pretrained_dipstat)
            print("P-value:", pretrained_p_value)

            print('CUSTOM')
            custom_scores = raw_predictions_custom[raw_predictions_custom['label_predicted'] == label]
            custom_dipstat, custom_p_value = diptest(custom_scores['logit'])
            print("Dip statistic:", custom_dipstat)
            print("P-value:", custom_p_value)

            if plot_score_distributions:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                fig.suptitle(label)
                ax1.hist(pretrained_scores['confidence'], bins=20, color='red', alpha=0.5)
                ax1.hist(custom_scores['confidence'], bins=20, color='blue', alpha=0.5)
                ax1.set_ylim(0, 25)
                ax1.set_title('Detection Scores')
                ax2.hist(pretrained_scores['logit'], bins=50, color='red', alpha=0.5)
                ax2.hist(custom_scores['logit'], bins=50, color='blue', alpha=0.5)
                ax2.set_xlim(-16, 16)
                ax2.set_title('Logit Scores')
                plt.tight_layout()
                plt.show()

        # # TODO: Sample present and absent examples equally to compare distributions
        # num_rows_present = raw_predictions_custom[raw_predictions_custom['label_truth'] == label].shape[0]
        # df_present = raw_predictions_custom[raw_predictions_custom['label_truth'] == label].sample(n=num_rows_present, random_state=1)
        # df_absent = raw_predictions_custom[raw_predictions_custom['label_truth'] != label].sample(n=num_rows_present, random_state=1)
        # custom_sampled_dipstat, custom_sampled_p_value = diptest(df_absent['logit'] + df_present['logit'])
        # print("Dip statistic:", custom_sampled_dipstat)
        # print("P-value:", custom_sampled_p_value)

        # if plot_score_distributions:
        #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        #     fig.suptitle(label)
        #     ax1.hist(df_absent['confidence'], bins=20, color='orange', alpha=0.5)
        #     ax1.hist(df_present['confidence'], bins=20, color='green', alpha=0.5)
        #     ax1.set_ylim(0, 25)
        #     ax1.set_title('Detection Scores')
        #     ax2.hist(df_absent['logit'], bins=20, color='orange', alpha=0.5)
        #     ax2.hist(df_present['logit'], bins=20, color='green', alpha=0.5)
        #     ax2.set_xlim(-16, 16)
        #     ax2.set_title('Logit Scores')
        #     plt.tight_layout()
        #     plt.show()
    
    # TODO: Test Silverman's test for modality
    # from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott, select_bandwidth
    # silverman_bandwidth = bw_silverman(data)
    # # select bandwidth allows to set a different kernel
    # silverman_bandwidth_gauss = select_bandwidth(data, bw = 'silverman', kernel = 'gauss')
