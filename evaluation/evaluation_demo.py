# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
# The output of the model is a logistic regression score
# We need to transform this into a discrete binary classification (e.g. Robin present vs Robin not present)
# We can do this with a classification threshold.

# Precision
# Precision = TP/(TP + FP)
# When the model said "positive" class, was it right?

# True Positive Rate (a.k.a. "recall" or "sensitivity")
# TPR = TP/(TP + FN)
# Out of all possible possitives, how many did the model correctly identify?

# Precision matters when the cost of false positives is high; when misclassifying as present has serious consequences.
# High precision relates to a low false positive rate.
# Recall matters when the cost of false negatives is high; when misclassifying as absent has serious consequences.
# High recall relates to a low false negative rate.

# Precision and recall are well-defined when we've decided on a classification threshold. But what if we don't know what threshold to use?

# False Positive Rate (a.k.a. "fall-out" or "false alarm ratio")
# FPR = FP/(FP + TN)

# An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds.
# This curve plots TPR against FPR at different classification thresholds.
# Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives.

# AUC stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).
# AUC provides an aggregate measure of performance across all possible classification thresholds.
# One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example.
# AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.

# -------------------------------------------------
# Aggregate all selection tables

dir_in = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020'
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, PrecisionRecallDisplay, precision_recall_curve, auc, roc_curve, roc_auc_score, classification_report

# Find all selection table files
def find_files(directory, filetype):
    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(filetype):
                results.append(os.path.join(root, file))
    return results
selection_tables = find_files(dir_in, '.txt')

import re

def get_species_and_confidence_from_file(file):
    # print('get_species_and_confidence_from_file')
    # Regular expression pattern to extract species name and confidence from filename
    pattern = re.compile(r'(.+)-(\d+\.\d+)_') # e.g. 'Barred Owl-0.9557572603225708_SMA...'
    match = pattern.match(file)
    if match:
        species = match.group(1)
        confidence = float(match.group(2))
        # print("File:", file)
        # print("Species:", species)
        # print("Confidence:", confidence)
        # print()  # for separating the outputs
        return species, confidence
    else:
        print("Unable to determine species and/or confidence from file:", file)

# Combine selection table files into a single 'annotations' dataframe
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

    if table.isna().any().any():
        print(f'WARNING: {table_file} contains NaN values')
        table = table.dropna()

    # Parse the species names and confidence scores from 'Begin File' column
    species_and_confidences = list(map(lambda f: get_species_and_confidence_from_file(f), table['begin file']))
    # Separate the tuples into two separate lists
    species, confidences = zip(*species_and_confidences)

    table.rename(columns={'species': 'label_truth'}, inplace=True) # Rename 'species' to 'label_truth', i.e. annotated species or sound label
    table.insert(0, 'species_predicted', species) # Add new column for species predicted by the classifier
    table['species_predicted'] = table['species_predicted'].str.lower()
    table.insert(2, 'confidence', confidences)

    # Clean up species names
    table['species_predicted'] = table['species_predicted'].str.replace('_', "'")
    table['label_truth'] = table['label_truth'].str.replace('_', "'")

    table.rename(columns={'begin file': 'file'}, inplace=True) # Rename 'begin file' to 'file'
    # table['file'] = os.path.dirname(table_file) + '/' + table['file'] # Reconstruct full path to associated audio data

    annotations = pd.concat([annotations, table], ignore_index=True)

print(annotations)

# More clean up of typos
annotations.loc[annotations['label_truth'].str.contains('not|non', case=False), 'label_truth'] = '0' # a value of 0 indicates the species_predicted is NOT present
annotations['label_truth'].replace(['unknown', 'unkown'], 'unknown', inplace=True)

unique_labels = sorted(annotations['label_truth'].unique()) # Get a sorted list of unique species names
print(f'{len(unique_labels)} unique labels annotated: {unique_labels}')

# -------------------------------------------------
# DEMO: Look at a specific species
species_name = 'barred owl'

# Filter for only barred owl annotations
annotations_species = annotations[annotations['species_predicted'] == species_name]

# Exclude files with an "unknown" label_truth from consideration
annotations_species = annotations_species.groupby('file').filter(lambda x: 'unknown' not in x['label_truth'].values)

# Sort by confidence
annotations_species = annotations_species.sort_values(by='confidence', ascending=True)

print(annotations_species)

# plt.figure(1, figsize=(4, 3))
# plt.scatter(labels_binary['confidence'], labels_binary['label_truth'])
# plt.title(species_name)
# plt.ylabel('presence')
# plt.xlabel('confidence')
# plt.show()

# Evaluate performance at different thresholds
perf_thresholds = pd.DataFrame()
for threshold in np.linspace(0.0, 1.0, num=20, endpoint=False):
    print(f'Evaluating threshold {threshold}')

    annotations_species['label_predicted'] = annotations_species['confidence'].apply(lambda x: 1 if x >= threshold else 0) # Convert to binary presence/absence with threshold

    # Determine if species was truly present (1) or not (0) for each unique file
    labels_binary = annotations_species.groupby('file').apply(lambda group: 1 if species_name in group['label_truth'].values else 0).reset_index()
    labels_binary.columns = ['file', 'label_truth']
    labels_binary = pd.merge(
        labels_binary,
        annotations_species[['file', 'label_predicted', 'confidence']].drop_duplicates(), # Get label_predicted for each unique file
        on='file',
        how='left'
    )
    # print(labels_binary)

    # Generate confusion matrix and retrieve true negative, false positive, false negative, and true positive count
    conf_mtx = confusion_matrix(labels_binary['label_truth'], labels_binary['label_predicted'])
    tn, fp, fn, tp = conf_mtx.ravel()
    # disp = ConfusionMatrixDisplay(confusion_matrix)
    # disp.plot()
    # plt.show()

    # Calculate precision and recall
    precision = precision_score(labels_binary['label_truth'], labels_binary['label_predicted']) # TP/(TP + FP)
    recall =    recall_score(labels_binary['label_truth'], labels_binary['label_predicted']) # TP/(TP + FN)

    print("Precision:", precision)
    print("Recall:", recall)

    row = {'threshold': threshold, 'precision': precision, 'recall': recall}
    perf_thresholds = pd.concat([perf_thresholds, pd.DataFrame([row])], ignore_index=True)

print(perf_thresholds)

# Plot precision and recall as a function of threshold
precision, recall, th = precision_recall_curve(labels_binary['label_truth'], labels_binary['confidence'])
print(f'th {th}')
plt.plot(th, precision[:-1], label='Precision', marker='.') 
plt.plot(th, recall[:-1], label='Recall', marker='.') 
plt.xlabel('threshold') 
plt.ylabel('performance')
plt.title(f'Threshold Performance: {species_name} (N={len(labels_binary)})')
plt.legend() 
plt.show()

# Calculate the precision-recall Area Under the Curve (AUC)
# This score can then be used as a point of comparison between different models on a binary classification problem where a score of 1.0 represents a model with perfect skill.
pr_auc = auc(recall, precision)

# Plot precision-recall curve
no_skill = len(labels_binary[labels_binary['label_truth']==1]) / len(labels_binary)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Baseline', color='gray')
plt.plot(recall, precision, marker='.', label='Classifier')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.title(f'Precision-Recall (AUC {pr_auc:.2f}): {species_name} (N={len(labels_binary)})')
plt.legend()
plt.show()

# ROC
# The ROC curve is a fundamental tool for evaluating binary classification models. It displays the relationship between the true positive rate (TPR) and the false positive rate (FPR) as we vary the discrimination threshold. In simpler terms, it helps us understand how our model performs at different levels of certainty.
# The ROC Area Under the Curve (ROC-AUC) serves as a handy summary statistic of the ROC curve. The AUC tells us about the probability that a randomly selected positive instance ranks higher than a randomly selected negative one. If the AUC equals 1, we have a perfect model. If the AUC equals 0.5, our model is no better than a coin flip.
fpr, tpr, th = roc_curve(labels_binary['label_truth'], labels_binary['confidence'], drop_intermediate=False)

roc_auc = roc_auc_score(labels_binary['label_truth'], labels_binary['confidence'])

ns_probs = [0 for _ in range(len(labels_binary))] # no skill classifier that only predicts 0 for all examples
ns_roc_auc_score = roc_auc_score(labels_binary['label_truth'], ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(labels_binary['label_truth'], ns_probs, drop_intermediate=False)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Baseline', color='gray')
plt.plot(fpr, tpr, marker='.', label='Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title(f'ROC (AUC {roc_auc:.2f}): {species_name} (N={len(labels_binary)})')
plt.legend()
plt.show()