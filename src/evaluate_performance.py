##
# Demo script to validate evaluation data and evaluate classifier performance for each species class
# Requires annotations from Google Drive. Ensure these are downloaded locally before running.
#
# For a given audio segment, the classifier outputs a logistic regression score for each
# species class representing its confidence that the species is present in the segment.
# We need to transform this continuous confidence score into a discrete binary classification
# (e.g. Robin present vs Robin not present). We can do this with a classification threshold.
#
# TODO: Load all raw classifier detections. This is needed in future for evaluating alternate classifiers one-to-one, or to use non-species-specific detections for evaluation
# TODO: Determine the most commonly-confused (associated) labels per species for false positives.

# Declare some configuration variables ------------------------------------

# Root directory for the annotations data
# Set this to a specific directory to evaluate performance on a subset of the data
dir_annotations = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment4/SMA00486_20200523'

# TODO: Root directory for the raw detection data
# dir_detections = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/raw_detections/2020'

# Evaluate all species classes...
species_to_evaluate = 'all'
# ...or look at just a few
# species_to_evaluate = ['american robin', 'common raven', 'band-tailed pigeon', 'barred owl', 'american kestrel']

plot = False # Plot the results

# TODO: Once all annotations are complete and every detection has been evaluated, set this to False
only_annotated = True # DEBUG: skip files that do not have annotations (selection tables)

# Load required packages -------------------------------------------------
import sys
import os                       # File navigation
import pandas as pd             # Data manipulation
import matplotlib.pyplot as plt # Plotting
import sklearn.metrics          # Classifier evaluation
import re                       # Regular expressions
import numpy as np              # Mathematics

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

## COLLATE ANNOTATION AND DETECTION DATA =================================================

# Declare 'evaluated_detections' - Get the list of files (detections) that were actually evaluated by annotators
# These are all the .wav files under each site-date folder in e.g. 'annotation/data/_annotator/2020/Deployment4/SMA00410_20200523'
evaluated_detection_files_paths = find_files(dir_annotations, '.wav')
evaluated_detection_files = list(map(os.path.basename, evaluated_detection_files_paths))
# Parse the species names and confidence scores from the .wav filenames
file_metadata = list(map(lambda f: parse_metadata_from_file(f), evaluated_detection_files))
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
selection_tables = find_files(dir_annotations, '.txt')
raw_annotations = pd.DataFrame()
print('Loading annotation selection tables...')
for table_file in sorted(selection_tables):
    # print(f'Loading file {os.path.basename(table_file)}...')

    table = pd.read_csv(table_file, sep='\t') # Load the file as a dataframe

    # Clean up data by normalizing column names and species values to lowercase
    table.columns = table.columns.str.lower()
    table['species'] = table['species'].str.lower()

    # Check the validity of the table
    cols_needed = ['species', 'begin file', 'file offset (s)', 'delta time (s)']
    if not all(col in table.columns for col in cols_needed):
        missing_columns = [col for col in cols_needed if col not in table.columns]
        print_warning(f"Missing columns {missing_columns} in {os.path.basename(table_file)}. Skipping...")
        continue

    if table.empty:
        print_warning(f'{table_file} has no selections. Skipping...')
        continue

    table = table[cols_needed] # Discard unnecessary data

    if table.isna().any().any():
        print_warning(f'{table_file} contains NaN values. Skipping...')
        continue

    # Parse the species names and confidence scores from 'Begin File' column
    file_metadata = list(map(lambda f: parse_metadata_from_file(f), table['begin file']))
    species, confidences, serialnos, dates, times = zip(*file_metadata)

    table.rename(columns={'species': 'label_truth'}, inplace=True) # Rename 'species' to 'label_truth'
    table.insert(0, 'species_predicted', species) # Add column for species predicted by the classifier
    # table.insert(2, 'confidence', confidences) # Add column for confidence values
    # table['serialno'] = serialnos

    # Clean up species names
    table['species_predicted'] = table['species_predicted'].str.lower()
    table['species_predicted'] = table['species_predicted'].str.replace('_', "'")
    table['label_truth'] = table['label_truth'].str.replace('_', "'")

    table.rename(columns={'begin file': 'file'}, inplace=True) # Rename 'begin file' to 'file'

    raw_annotations = pd.concat([raw_annotations, table], ignore_index=True) # Store the table

# More clean up of typos
raw_annotations.loc[raw_annotations['label_truth'].str.contains('not|non', case=False), 'label_truth'] = '0' # 0 indicates the species_predicted is NOT present
raw_annotations['label_truth'] = raw_annotations['label_truth'].replace(['unknown', 'unkown'], 'unknown')

# Throw warning if missing annotations for any species in species_predicted
evaluated_species = sorted(evaluated_detections['species_predicted'].unique())
annotated_species = sorted(raw_annotations['species_predicted'].unique())
missing_species = [species for species in evaluated_species if species not in annotated_species]
if (len(missing_species) > 0):
    print_warning(f'Missing annotations for the following species:\n{missing_species}')

# Throw warning for unique non-species labels
species_classes = pd.read_csv('/Users/giojacuzzi/repos/avian-bioacoustics-oesf/src/classification/species_list/species_list_OESF.txt', header=None) # Get list of all species
species_classes = [name.split('_')[1].lower() for name in species_classes[0]]
unique_labels = sorted(raw_annotations['label_truth'].unique()) # Get a sorted list of unique species names
unique_labels = [label for label in unique_labels if label not in species_classes]
print_warning(f'{len(unique_labels)} unique non-species labels annotated: {unique_labels}')

sys.exit()

if species_to_evaluate == 'all':
    species_to_evaluate = sorted(species_classes)

# print(raw_annotations)
    
performance_metrics = pd.DataFrame() # Container for performance metrics of all species

# For each unique species (or a particular species), compute data labels and evaluate performance
for species in species_to_evaluate:

    ## COLLATE SPECIES DATA ===============================================================
    print(f'Collating "{species}" data...')
    if (species in missing_species):
        print_warning(f'No annotations for {species}')

    # Get all annotations for the species
    if True:
        # For now, only include detections (TP and FP).
        detection_labels = raw_annotations[raw_annotations['species_predicted'] == species]
    else:
        # TODO: Include not only detections (TP and FP), but also non-detections (FN).
        detection_labels = raw_annotations[(raw_annotations['species_predicted'] == species) | (raw_annotations['label_truth'] == species_to_evaluate)]
        # NOTE: The resulting confidence scores are incorrect where species_predicted != species_name
        detection_labels.loc[detection_labels['species_predicted'] != species, 'confidence'] = np.nan # Set confidence to NaN for the identified rows
        # TODO: Instead of getting confidence score from the detection file for TP and FP, get ALL confidence scores from the raw data (TP, FP, and FN). Use this to overwrite 'confidence'?
    detection_labels = detection_labels.sort_values(by='file')
    # print('All annotations:')
    # print(detection_labels)

    # Include any evaluated detections for the species without annotations (which should all be FP)
    detection_labels = pd.merge(
        evaluated_detections[evaluated_detections['species_predicted'] == species],
        detection_labels,
        how='left' # 'left' to include all evaluated detections, regardless of whether they have annotations.
        # 'inner' would only include those detections that have annotations.
    )
    detection_labels = detection_labels.sort_values(by='file')

    if detection_labels.empty:
        print_warning(f'No samples for {species}. Skipping...')
        continue

    if detection_labels['label_truth'].isna().any():
        print_warning('Assuming missing (NaN) label_truth values due to lack of annotation selections are false positives (0)...')
        if only_annotated:
            detection_labels.dropna(subset=['label_truth'], inplace=True)
            if detection_labels.empty:
                print_warning(f'No remaining samples for {species}. Skipping...')
                continue
        else:
            detection_labels['label_truth'] = detection_labels['label_truth'].fillna(0)

    # TODO: When including non-detections (FN), do we need to pull confidence scores here instead? For the NaN values?

    # print('All annotations and samples:')
    # print(detection_labels)

    # For each unique file (i.e. detection), determine if the species was present (1), unknown (?), or not (0) among all annotation labels
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
        detection_labels.drop(columns=['label_truth', 'file offset (s)', 'delta time (s)']), # drop selection-specific fields that no longer apply
        detection_labels.groupby('file').apply(compute_label_presence, label=species).reset_index(name='label_truth'), # label_truth here will replace the previous label_truth
        on='file'
    )
    # Drop duplicate rows based on the 'file' column
    detection_labels = detection_labels.drop_duplicates(subset='file')
    print('Species labels by file:')
    print(detection_labels)

    ## EVALUATE SPECIES PERFORMANCE =========================================================

    # Exclude files with an "unknown" label_truth from consideration
    n_unknown = (detection_labels['label_truth'] == 'unknown').sum()
    if n_unknown > 0:
        print_warning(f"{n_unknown} samples with unknown species. Excluding these from consideration...")
        detection_labels = detection_labels[detection_labels['label_truth'] != 'unknown']

    if detection_labels.empty:
        print_warning(f'No remaining samples for {species}. Skipping...')
        continue

    # print(detection_labels)


    # Precision is the proportion of true positives among positive predictions, TP/(TP + FP). Intuitively,
    # when the model says "Barred Owl", how often is it correct? Precision matters when the cost of false
    # positives is high; when misclassifying as present has serious consequences.
    #
    # Recall is the proportion of true positives identified out of all actual positives, TP/(TP + FN), also
    # called the "true positive rate" or "sensitivity". Intuitively, out of all actual Barred Owl calls, how
    # many did the model correctly identify? Recall matters when the cost of false negatives is high; when
    # misclassifying as absent has serious consequences. High recall relates to a low false negative rate.
    #
    # The Precision-Recall curve summarizes the tradeoff between precision and recall as we vary the
    # confidence threshold. This relationship is only concerned with the correct prediction of the species
    # label, and does not require true negatives to be calculated. As such, Precision-Recall curves are
    # appropriate when the evaluation data are imbalanced between each class (i.e. present vs non-present).
    # This means is that a large number of negative instances won’t skew our understanding of how well our model
    # performs on the positive species class.
    precision, recall, th = sklearn.metrics.precision_recall_curve(detection_labels['label_truth'], detection_labels['confidence'], pos_label=species)

    if plot:
        # Plot precision and recall as a function of threshold
        plt.plot(th, precision[:-1], label='Precision', marker='.') 
        plt.plot(th, recall[:-1], label='Recall', marker='.')
        plt.xlabel('threshold') 
        plt.ylabel('performance')
        plt.title(f'Threshold Performance: {species} (N={len(detection_labels)})')
        plt.legend() 
        plt.show()

    # The area under the precision-recall curve, AUC-PR, is a useful summary statistic of the the relationship.
    # The AUC-PR ranges from 0.0 to 1.0, with 1.0 indicating a perfect classifier. However, unlike the AUC
    # of the ROC curve, the AUC-PR of a random (baseline) classifier is equal to the proportion of positive
    # instances in the dataset. This makes it a more conservative (and in many cases, more realistic) metric
    # when dealing with imbalanced data. Note that because this is calculated using the interpolated trapezoidal
    # rule, it can lead to an overestimated measurement of performance
    pr_auc = sklearn.metrics.auc(recall, precision)

    # The average precision from prediction scores (AP) summarizes a precision-recall curve as the weighted mean of
    # precisions achieved at each threshold, with the increase in recall from the previous threshold used as the
    # weight. This can provide a more realistic measure of performance than AUC as it is not interpolated among scores.
    pr_ap = sklearn.metrics.average_precision_score(detection_labels['label_truth'], detection_labels['confidence'], pos_label=species)

    # A baseline or "unskilled" classifier is one that cannot discriminate between the classes and would
    # predict a random class or a constant class in all cases. This is represented by a horizontal line
    # at the ratio of positive to negative examples in the dataset.
    no_skill = len(detection_labels[detection_labels['label_truth']==species]) / len(detection_labels)

    if plot:
        # Plot precision-recall curve
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Baseline', color='gray')
        plt.plot(recall, precision, marker='.', label='Classifier')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall (AUC {pr_auc:.2f}, AP {pr_ap:.2f}): {species} (N={len(detection_labels)})')
        plt.legend()
        plt.show()

    # Store the performance metrics
    performance_metrics = pd.concat([performance_metrics, pd.DataFrame({
        'species': [species],
        'AUC-PR':  [round(pr_auc, 2)],                      # Precision-Recall AUC
        'AP':      [round(pr_ap, 2)],                      # Average precision
        'p_mean':  [round(precision.mean(),2)],             # Average precision across all examples
        'N':       [len(detection_labels)],                    # Total number of examples
        'N_1':     [sum(detection_labels['label_truth'] == species)], # Total number of positive examples
        'N_U':     [n_unknown] # Total number of unknown examples excluded from evaluation
    })], ignore_index=True)

    # TODO: Use F1-score to determine an optimal threshold, e.g. https://ploomber.io/blog/threshold/

    # # NOTE: ROC EVALUATIONS SHOWN FOR DEMONSTRATION. IMBALANCED EVALUATION DATA PRECLUDES INFERENCE.
    # # The Receiver Operating Characteristic curve summarizes the tradeoff between the true positive rate (i.e. recall)
    # # and the false positive rate (FPR) as we vary the confidence threshold. ROC curves are appropriate when the
    # # evaluation data are balanced between each class (i.e. present vs non-present).
    # fpr, tpr, th = sklearn.metrics.roc_curve(labels_binary['label_truth'], labels_binary['confidence'], drop_intermediate=False)

    # # The ROC Area Under the Curve (ROC-AUC) is a useful summary statistic of the ROC curve that tells us the
    # # probability that a randomly-selected positive example ranks higher than a randomly-selected negative one.
    # # ROC-AUC == 1 if our model is perfect, while ROC-AUC == 0.5 if our model is no better than a coin flip.
    # # Note that a low ROC-AUC can also reflect a class imbalance in the evaluation data.
    # roc_auc = sklearn.metrics.roc_auc_score(labels_binary['label_truth'], labels_binary['confidence'])

    # # Plot ROC
    # ns_probs = [0 for _ in range(len(labels_binary))] # no skill classifier that only predicts 0 for all examples
    # ns_roc_auc_score = sklearn.metrics.roc_auc_score(labels_binary['label_truth'], ns_probs)
    # ns_fpr, ns_tpr, _ = sklearn.metrics.roc_curve(labels_binary['label_truth'], ns_probs, drop_intermediate=False)
    # plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Baseline', color='gray')
    # plt.plot(fpr, tpr, marker='.', label='Classifier')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate (Recall)')
    # plt.title(f'ROC (AUC {roc_auc:.2f}): {species_name} (N={len(labels_binary)})')
    # plt.legend()
    # plt.show()

# Sort and print the performance metrics
performance_metrics = performance_metrics.sort_values(by='p_mean', ascending=False).reset_index(drop=True)
print(performance_metrics.to_string())
