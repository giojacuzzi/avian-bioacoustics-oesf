# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
# The output of the model is a logistic regression score
# We need to transform this into a discrete binary classification (e.g. Robin present vs Robin not present)
# We can do this with a classification threshold.

# Load required packages
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
import sklearn.metrics
import re

# Utility functions -------------------------------------------------

# Function to parse species name and confidence from filename
def get_species_and_confidence_from_file(file):
    pattern = re.compile(r'(.+)-(\d+\.\d+)_') # Regular expression, e.g. 'Barred Owl-0.9557572603225708_SMA...'
    match = pattern.match(file)
    if match:
        species = match.group(1)
        confidence = float(match.group(2))
        return species, confidence
    else:
        print("Unable to determine species and/or confidence from file:", file)

# Find all selection table files under a root directory
def find_files(directory, filetype):
    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(filetype):
                results.append(os.path.join(root, file))
    return results

# Load and aggregate evaluation data -------------------------------------------------
        
# Locate all selection tables
dir_in = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020'
selection_tables = find_files(dir_in, '.txt')

# Load selection table files and combine into a single 'annotations' dataframe
annotations = pd.DataFrame()   
for table_file in selection_tables:
    print(f'Loading file {os.path.basename(table_file)}...')

    table = pd.read_csv(table_file, sep='\t') # Load the file as a dataframe

    # Clean up data by normalizing column names and species values to lowercase
    table.columns = table.columns.str.lower()
    table['species'] = table['species'].str.lower()

    # Check the validity of the table
    cols_needed = ['species', 'begin file', 'file offset (s)', 'delta time (s)']
    if not all(col in table.columns for col in cols_needed):
        missing_columns = [col for col in cols_needed if col not in table.columns]
        print(f"WARNING: {table_file} is missing columns {missing_columns}")
        continue
    if table.empty:
        print(f'{table_file} has no selections. Continuing...')
        continue

    table = table[cols_needed] # Discard unnecessary data

    # Drop any empty tables
    # TODO: An entry should be fabricated for each file associated with the table as a valid negative example
    if table.isna().any().any():
        print(f'WARNING: {table_file} contains NaN values')
        table = table.dropna()

    # Parse the species names and confidence scores from 'Begin File' column
    species_and_confidences = list(map(lambda f: get_species_and_confidence_from_file(f), table['begin file']))
    species, confidences = zip(*species_and_confidences)

    table.rename(columns={'species': 'label_truth'}, inplace=True) # Rename 'species' to 'label_truth'
    table.insert(0, 'species_predicted', species) # Add column for species predicted by the classifier
    table.insert(2, 'confidence', confidences) # Add column for confidence values

    # Clean up species names
    table['species_predicted'] = table['species_predicted'].str.lower()
    table['species_predicted'] = table['species_predicted'].str.replace('_', "'")
    table['label_truth'] = table['label_truth'].str.replace('_', "'")

    table.rename(columns={'begin file': 'file'}, inplace=True) # Rename 'begin file' to 'file'

    annotations = pd.concat([annotations, table], ignore_index=True) # Store the table

print(annotations)

# More clean up of typos
annotations.loc[annotations['label_truth'].str.contains('not|non', case=False), 'label_truth'] = '0' # a value of 0 indicates the species_predicted is NOT present
annotations['label_truth'].replace(['unknown', 'unkown'], 'unknown', inplace=True)

unique_labels = sorted(annotations['label_truth'].unique()) # Get a sorted list of unique species names
print(f'{len(unique_labels)} unique labels annotated: {unique_labels}')

# Evaluate model performance on species classes -------------------------------------------------
# species_to_evaluate = sorted(annotations['species_predicted'].unique()) # Evaluate all species classes...
species_to_evaluate = ['band-tailed pigeon'] # ...or look at just a few

for species_name in species_to_evaluate:
    print(f'Evaluating class "{species_name}"...')

    # Filter for only examples predicted to be the species
    # TODO: In the future, include ALL examples. This requires pulling confidence scores from the raw data,
    # rather than the detection annotations
    annotations_species = annotations[annotations['species_predicted'] == species_name]

    # Exclude files with an "unknown" label_truth from consideration
    annotations_species = annotations_species.groupby('file').filter(lambda x: 'unknown' not in x['label_truth'].values)

    # Sort by confidence
    annotations_species = annotations_species.sort_values(by='confidence', ascending=True)
    print(annotations_species)

    # For each unique file (i.e. detection), determine if the species was truly present (1) or not (0) among all labels
    labels_binary = annotations_species.groupby('file').apply(lambda group: 1 if species_name in group['label_truth'].values else 0).reset_index()
    labels_binary.columns = ['file', 'label_truth']
    labels_binary = pd.merge(
        labels_binary,
        annotations_species[['file', 'confidence']].drop_duplicates(), # Get the detection confidence associated with this file
        on='file',
        how='left'
    )
    print(labels_binary)

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
    precision, recall, th = sklearn.metrics.precision_recall_curve(labels_binary['label_truth'], labels_binary['confidence'])

    # Plot precision and recall as a function of threshold
    plt.plot(th, precision[:-1], label='Precision', marker='.') 
    plt.plot(th, recall[:-1], label='Recall', marker='.')
    plt.xlabel('threshold') 
    plt.ylabel('performance')
    plt.title(f'Threshold Performance: {species_name} (N={len(labels_binary)})')
    plt.legend() 
    plt.show()

    # The area under the precision-recall curve, AUC-PR, is a useful summary statistic of the the relationship.
    # The AUC-PR ranges from 0.0 to 1.0, with 1.0 indicating a perfect classifier. However, unlike the AUC
    # of the ROC curve, the AUC-PR of a random (baseline) classifier is equal to the proportion of positive
    # instances in the dataset. This makes it a more conservative (and in many cases, more realistic) metric
    # when dealing with imbalanced data.
    pr_auc = sklearn.metrics.auc(recall, precision)

    # A baseline or "unskilled" classifier is one that cannot discriminate between the classes and would
    # predict a random class or a constant class in all cases. This is represented by a horizontal line
    # at the ratio of positive to negative examples in the dataset.
    no_skill = len(labels_binary[labels_binary['label_truth']==1]) / len(labels_binary)

    # Plot precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Baseline', color='gray')
    plt.plot(recall, precision, marker='.', label='Classifier')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall (AUC {pr_auc:.2f}): {species_name} (N={len(labels_binary)})')
    plt.legend()
    plt.show()

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
