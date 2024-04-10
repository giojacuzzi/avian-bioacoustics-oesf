##
# Demo script to evaluate classifier performance for each species class
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

# Annotation directories to use as evaluation data
dirs = [
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment4/SMA00410_20200523', # Jack
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment4/SMA00424_20200521', # Stevan
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment4/SMA00486_20200523', # Summer
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment4/SMA00556_20200524', # Gio
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment6/SMA00556_20200618', # Stevan
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment6/SMA00349_20200619', # Mirella
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment6/SMA00399_20200619', # Jessica
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment8/SMA00346_20200716', # Summer
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment8/SMA00339_20200717',  # Jack
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment2/SMA00351_20200424_Data', # Mirella
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment8/SMA00370_20200716', # Jessica
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment8/SMA00424_20200717', # Iris
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment6/SMA00404_20200618', # Iris
]

# Evaluate all species classes...
species_to_evaluate = 'all'
# ...or look at just a few
# species_to_evaluate = ["varied thrush", "pacific wren", "evening grosbeak", "hairy woodpecker", "golden-crowned kinglet", "barred owl", "chestnut-backed chickadee", "macgillivray's warbler", "townsend's warbler"]

plot = False # Plot the results

# TODO: Once all annotations are complete and every detection has been evaluated, set this to False
only_annotated = False # DEBUG: skip files that do not have annotations (selection tables)

# Load required packages -------------------------------------------------
import pandas as pd             # Data manipulation
import matplotlib.pyplot as plt # Plotting
import sklearn.metrics          # Classifier evaluation
import numpy as np              # Mathematics
from annotation.annotations import *
from utils.log import *

# Collate annotation data
raw_annotations = collate_annotations(dirs = dirs, overwrite = True)

species = labels.get_species_classes()
if species_to_evaluate == 'all':
    species_to_evaluate = sorted(species)

# Warn if missing annotations for any species in species_predicted
species_annotated = [label for label in sorted(raw_annotations['label_truth'].astype(str).unique()) if label in species]
print(f'Retrieved {len(raw_annotations)} total annotations for {len(species_annotated)}/{len(species)} species classes')
if len(species_annotated) < len(species):
    print_warning(f'Missing positive annotations for species: {[label for label in species if label not in species_annotated]}')

performance_metrics = pd.DataFrame() # Container for performance metrics of all species

absent_metrics = pd.DataFrame() # Container for metrics of species that are absent (no positive examples)

# For each unique species (or a particular species), compute data labels and evaluate performance
plots = []

for i, species in enumerate(species_to_evaluate):

    ## COLLATE SPECIES DATA ===============================================================
    print(f'Collating "{species}" data...')

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
        detection_labels.groupby('file').apply(compute_label_presence, include_groups=False, label=species).reset_index(name='label_truth'), # label_truth here will replace the previous label_truth
        on='file'
    )
    # Drop duplicate rows based on the 'file' column
    detection_labels = detection_labels.drop_duplicates(subset='file')
    # print('Species labels by file:')
    # print(detection_labels)
    n_examples = len(detection_labels)

    ## EVALUATE SPECIES PERFORMANCE =========================================================

    # Exclude files with an "unknown" label_truth from consideration
    n_unknown = (detection_labels['label_truth'] == 'unknown').sum()
    if n_unknown > 0:
        print_warning(f"{n_unknown} detections with unknown species. Excluding these from consideration...")
        detection_labels = detection_labels[detection_labels['label_truth'] != 'unknown']

    N_P = sum(detection_labels['label_truth'] == species) # Total number of positive examples
    N_N = sum(detection_labels['label_truth'] != species) # Total number of negative examples
    
    if len(detection_labels) == 0:
        print_error(f"No remaining known detections to evaluate for '{species}'. Skipping...")
        continue

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
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(detection_labels['label_truth'], detection_labels['confidence'], pos_label=species)

    padding = 0.01
    font_size = 9

    if plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        # Plot precision and recall as a function of threshold
        ax1.plot(thresholds, precision[:-1], label='Precision', marker='.') 
        ax1.plot(thresholds, recall[:-1], label='Recall', marker='.')
        ax1.set_xlabel('Threshold') 
        ax1.set_ylabel('Performance')
        ax1.set_title(f'Threshold performance', fontsize=font_size)
        ax1.set_xlim(0.0-padding, 1.0+padding)
        ax1.set_ylim(0.0-padding, 1.0+padding)
        ax1.legend(loc='lower left') 

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
        ax2.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Baseline', color='gray')
        ax2.plot(recall, precision, marker='.', label='Classifier')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall (AUC {pr_auc:.2f}, AP {pr_ap:.2f})', fontsize=font_size)
        ax2.set_xlim([0.0-padding, 1.0+padding])
        ax2.set_ylim([0.0-padding, 1.0+padding])
        ax2.legend(loc='lower left')

    # Store the performance metrics
    performance_metrics = pd.concat([performance_metrics, pd.DataFrame({
        'species':   [species],
        'AUC-PR':    [round(pr_auc, 2)],                                # Precision-Recall AUC
        'AP':        [round(pr_ap, 2)],                                 # Average precision
        'p_mean':    [round(precision.mean(),2)],                       # Average precision across all thresholds
        'p_max':     [round(precision[np.argmax(precision[:-1])],2)],   # Maximum precision across all thresholds
        'p_max_th':  [round(thresholds[np.argmax(precision[:-1])],2)],  # Score threshold to maximize precision
        'p_max_r':   [round(recall[np.argmax(precision[:-1])],2)],      # Recall at maximum precision using threshold
        'N':         [n_examples],                                      # Total number of examples
        'N_P':       [N_P],                                             # Total number of positive examples
        'N_N':       [N_N],                                             # Total number of negative examples
        'N_unknown': [n_unknown]                                        # Total number of unknown examples excluded from evaluation
    })], ignore_index=True)

    # TODO: Use F1-score to determine an optimal threshold, e.g. https://ploomber.io/blog/threshold/

    # NOTE: ROC EVALUATIONS SHOWN FOR DEMONSTRATION. IMBALANCED EVALUATION DATA PRECLUDES INFERENCE.
    # The Receiver Operating Characteristic curve summarizes the tradeoff between the true positive rate (i.e. recall)
    # and the false positive rate (FPR) as we vary the confidence threshold. ROC curves are appropriate when the
    # evaluation data are balanced between each class (i.e. present vs non-present).
    fpr, tpr, th = sklearn.metrics.roc_curve(detection_labels['label_truth'], detection_labels['confidence'], pos_label=species, drop_intermediate=False)

    # The ROC Area Under the Curve (ROC-AUC) is a useful summary statistic of the ROC curve that tells us the
    # probability that a randomly-selected positive example ranks higher than a randomly-selected negative one.
    # ROC-AUC == 1 if our model is perfect, while ROC-AUC == 0.5 if our model is no better than a coin flip.
    # Note that a low ROC-AUC can also reflect a class imbalance in the evaluation data.
    if N_P > 0 and N_N > 0:
        roc_auc = sklearn.metrics.roc_auc_score(detection_labels['label_truth'], detection_labels['confidence'])
    else:
        print_warning("Could not compute ROC AUC, no negative examples.")
        roc_auc = 0.0

    # Plot ROC
    if False:
        # ns_probs = [species for _ in range(len(detection_labels))] # no skill classifier that only predicts 1 for all examples
        # ns_roc_auc_score = sklearn.metrics.roc_auc_score(detection_labels['label_truth'], ns_probs, pos_label=species)
        # ns_fpr, ns_tpr, _ = sklearn.metrics.roc_curve(detection_labels['label_truth'], ns_probs, pos_label=species, drop_intermediate=False)
        # plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Baseline', color='gray')
        ax3.plot(fpr, tpr, marker='.', label='Classifier')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate (Recall)')
        ax3.set_title(f'ROC (AUC {roc_auc:.2f})', fontsize=font_size)
        ax3.set_xlim([0.0-padding, 1.0+padding])
        ax3.set_ylim([0.0-padding, 1.0+padding])
        ax3.legend(loc='lower right')

    if plot:
        fig.suptitle(species, x=0.0, y=1.0, horizontalalignment='left', verticalalignment='top', fontsize=12)
        plt.tight_layout()
        plots.append((fig, (ax1, ax2)))

# Sort and print the performance metrics
performance_metrics = performance_metrics.sort_values(by='p_mean', ascending=False).reset_index(drop=True)

max_confidence_per_species = raw_annotations.rename(columns={'species_predicted': 'species'}).groupby('species')['confidence'].max()
performance_metrics = pd.merge(performance_metrics, max_confidence_per_species, on='species', how='left')
performance_metrics = performance_metrics.rename(columns={'confidence': 'max_conf'})
potential_species = pd.read_csv('data/species/Species List - Potential species.csv', index_col=None, usecols=['Common_Name', 'Rarity'])
potential_species = potential_species.rename(columns={'Common_Name': 'species', 'Rarity': 'rarity'})
potential_species['species'] = potential_species['species'].str.lower()
performance_metrics = pd.merge(potential_species, performance_metrics, on='species', how='left')

cols_as_ints = ['N','N_P','N_N','N_unknown']
performance_metrics[cols_as_ints] = performance_metrics[cols_as_ints].fillna(0)
performance_metrics[cols_as_ints] = performance_metrics[cols_as_ints].astype(int)

# N POSITIVE EXAMPLES
print('SPECIES CLASS PERFORMANCE ====================================================')
sorted_df = performance_metrics.sort_values(by=['N_P', 'AUC-PR'], ascending=[False, True])
print(sorted_df.to_string(index=False))

# COMMON
print('COMMON SPECIES ===============================================================')
common_species = performance_metrics.loc[performance_metrics['rarity']=='C']
common_species = common_species.sort_values(by=['N_P', 'N_unknown', 'max_conf'], ascending=[False, False, False])
print(common_species.to_string(index=False))

# RARE
print('RARE SPECIES ===============================================================')
rare_species = performance_metrics.loc[performance_metrics['rarity']=='R']
rare_species = rare_species.sort_values(by=['N_P', 'N_unknown', 'max_conf'], ascending=[False, False, False])
print(rare_species.to_string(index=False))

# N MISSING
print('MISSING SPECIES ==============================================================')
min_N_P = 7 + 1
missing_species = performance_metrics.loc[performance_metrics['N_P']<min_N_P]
print_error(f'{len(missing_species)} species with less than {min_N_P} positive examples:\n{missing_species.to_string(index=False)}')

if plot:
    for fig, ax in plots:
        fig.show()
        plt.show()


# # Print "confused" or simultaneously present classes
# for i, species in enumerate(species_to_evaluate):
#     print(f'CONFUSED CLASSES FOR {species}:')
#     species_labels = (raw_annotations[raw_annotations['species_predicted'] == species])
#     species_labels = species_labels[(species_labels['label_truth'] != '0') & (species_labels['label_truth'] != species)]
#     print(species_labels['label_truth'].value_counts())
