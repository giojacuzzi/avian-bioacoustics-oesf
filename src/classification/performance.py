# Evaluate performance for a single class (species)

import pandas as pd             # Data manipulation
import matplotlib.pyplot as plt # Plotting
import sklearn.metrics          # Classifier evaluation
import numpy as np              # Mathematics
from utils.log import *
import os

# detection_labels - a dataframe of detections with columns 'label_truth' (where a positive presence is represented by the species class label) and 'confidence'
# species - the species class label, e.g. "american crow"
def evaluate_species_performance(detection_labels, species, plot, digits=3, title_label='', save_to_dir=''):
    
    plots = []

    # Exclude files with an "unknown" label_truth from consideration
    n_unknown = (detection_labels['label_truth'] == 'unknown').sum()
    if n_unknown > 0:
        print_warning(f"{n_unknown} detections with unknown species. Excluding these from consideration...")
        detection_labels = detection_labels[detection_labels['label_truth'] != 'unknown']

    n_examples = len(detection_labels)
    n_P = sum(detection_labels['label_truth'] == species) # Total number of positive examples
    n_N = sum(detection_labels['label_truth'] != species) # Total number of negative examples
    
    if len(detection_labels) == 0:
        print_error(f"No detections to evaluate for '{species}'.")
        return

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

    f1_scores = 2*recall*precision/(recall+precision)
    f1_optimal_threshold = thresholds[np.argmax(f1_scores)]
    f1_max = np.max(f1_scores)

    if save_to_dir != '':
        os.makedirs(save_to_dir, exist_ok=True)
        stats = pd.DataFrame({
            'threshold': thresholds,
            'precision': precision[:-1],
            'recall': recall[:-1]
        })
        stats = stats.sort_values(by=['threshold', 'precision', 'recall'])
        stats.to_csv(f'{save_to_dir}/{species}.csv', index=False)

    padding = 0.01
    font_size = 9

    if plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        # Plot precision and recall as a function of threshold
        ax1.plot(thresholds, precision[:-1], label='Precision', marker='.') #, marker='.' 
        ax1.plot(thresholds, recall[:-1], label='Recall', marker='.') # , marker='.'
        ax1.plot(thresholds, f1_scores[:-1], label='F1 Score') # , marker='.'
        ax1.set_xlabel('Threshold') 
        ax1.set_ylabel('Performance')
        ax1.set_title(f'{title_label}\n{species}\nThreshold performance', fontsize=font_size)
        ax1.set_xlim(0.0-padding, 1.0+padding)
        ax1.set_ylim(0.0-padding, 1.0+padding)
        ax1.set_box_aspect(1)
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
        ax2.plot(recall, precision, label='Classifier') #  marker='.',
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'{title_label}\n{species}\nPrecision-Recall (AUC {pr_auc:.2f}, AP {pr_ap:.2f})', fontsize=font_size)
        ax2.set_xlim([0.0-padding, 1.0+padding])
        ax2.set_ylim([0.0-padding, 1.0+padding])
        ax2.legend(loc='lower left')
        ax2.set_box_aspect(1)

    # Plot ROC
    if False:
        # NOTE: ROC EVALUATIONS SHOWN FOR DEMONSTRATION. IMBALANCED EVALUATION DATA PRECLUDES INFERENCE.
        # The Receiver Operating Characteristic curve summarizes the tradeoff between the true positive rate (i.e. recall)
        # and the false positive rate (FPR) as we vary the confidence threshold. ROC curves are appropriate when the
        # evaluation data are balanced between each class (i.e. present vs non-present).
        fpr, tpr, th = sklearn.metrics.roc_curve(detection_labels['label_truth'], detection_labels['confidence'], pos_label=species, drop_intermediate=False)

        # The ROC Area Under the Curve (ROC-AUC) is a useful summary statistic of the ROC curve that tells us the
        # probability that a randomly-selected positive example ranks higher than a randomly-selected negative one.
        # ROC-AUC == 1 if our model is perfect, while ROC-AUC == 0.5 if our model is no better than a coin flip.
        # Note that a low ROC-AUC can also reflect a class imbalance in the evaluation data.
        if n_P > 0 and n_N > 0:
            roc_auc = sklearn.metrics.roc_auc_score(detection_labels['label_truth'], detection_labels['confidence'])
        else:
            print_warning("Could not compute ROC AUC, no negative examples.")
            roc_auc = 0.0

        if plot:
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
        # fig.suptitle(species, x=0.0, y=1.0, horizontalalignment='left', verticalalignment='top', fontsize=12)
        plt.tight_layout()
        plots.append((fig, (ax1, ax2)))
    
    if plot:
        for fig, ax in plots:
            fig.show()
            plt.show(block=False)

    if 'sites' in detection_labels.columns:
        N_sites = detection_labels[detection_labels['label_truth'] == species]['site'].nunique(dropna=True)
    else:
        N_sites = np.nan
    
    # Return the performance metrics
    return pd.DataFrame({
        'label':   [species],
        'AUC-PR':    [round(pr_auc, digits)],                                # Precision-Recall AUC
        'AP':        [round(pr_ap, digits)],                                 # Average precision
        'p_mean':    [round(precision.mean(), digits)],                       # Average precision across all thresholds
        'p_max':     [round(precision[np.argmax(precision[:-1])],  digits)], # Maximum precision across all thresholds
        'p_max_th':  [round(thresholds[np.argmax(precision[:-1])], digits)], # Score threshold to maximize precision
        'p_max_r':   [round(recall[np.argmax(precision[:-1])],     digits)], # Recall at maximum precision using threshold
        'r_max':     [round(recall[np.argmax(precision[:-1])],     digits)], # Maximum recall across all thresholds (should be 1.0)
        'r_max_th':  [round(thresholds[len(recall) - 1 - np.argmax(recall[::-1])], digits)], # Score threshold to maximize recall (last value to also secondarily max precision)
        'r_max_p':   [round( precision[len(recall) - 1 - np.argmax(recall[::-1])], digits)], # Precision at maximum precision using threshold (last value to also secondarily max precision)
        'f1_max':    [f1_max],                                          # Maximum F1 score across all thresholds
        'f1_max_th': [f1_optimal_threshold],                            # Score threshold to maximize F1 score
        'N':         [n_examples],                                      # Total number of examples (not including "unknown" examples)
        'N_P':       [n_P],                                             # Total number of positive examples
        'N_N':       [n_N],                                             # Total number of negative examples
        'N_unknown': [n_unknown],                                        # Total number of unknown examples excluded from evaluation
        'class_ratio': [round(n_P / n_examples, 2)],                                       # Class balance ratio (0.5 is perfectly balanced, 0.0 only negative, 1.0 only positive)
        'N_sites': [N_sites] # Number of unique sites with true presences
    })

# Returns a dataframe containing a confusion matrix (TP, FP, FN, TN) and number of truly present/absent sites for a given species from a detection history
def get_site_level_confusion_matrix(species, detections, threshold, all_sites):
    # Filter for species detections, excluding unknown examples
    detections_species = detections[(detections['label_predicted'] == species) & (detections['label_truth'] != 'unknown')]

    # Sites truly present (based on TP detections)
    sites_present  = detections_species[(detections_species['label_truth'] == species)]["StationName"].unique()
    # print(f'Sites present ({len(sites_present)}): {sites_present}')

    # Sites truly absent
    sites_absent = np.setdiff1d(all_sites, sites_present)
    # print(f'Sites absent  ({ len(sites_absent)}): {sites_absent}')

    if len(sites_present) + len(sites_absent) != len(all_sites):
        print_error(f'Number of sites present ({len(sites_present)}) and absent ({len(sites_absent)}) does not equal total number of sites ({len(all_sites)})')

    # Sites detected using the threshold
    detections_thresholded = detections_species[(detections_species['confidence'] >= threshold)]
    sites_detected = detections_thresholded["StationName"].unique()
    # print(f'Sites detected with threshold {threshold}  ({len(sites_detected)}): {sites_detected}')

    # Sites not detected using the threshold
    sites_notdetected = np.setdiff1d(all_sites, sites_detected)
    # print(f'Sites not detected with threshold {threshold}  ({len(sites_notdetected)}): {sites_notdetected}')

    if len(sites_detected) + len(sites_notdetected) != len(all_sites):
        print_error(f'Number of sites detected ({len(sites_detected)}) and not detected ({len(sites_notdetected)}) does not equal total number of sites ({len(all_sites)})')

    # TP - Number of sites correctly detected (at least once)
    tp_sites = np.intersect1d(sites_present, sites_detected)
    nsites_tp = len(tp_sites)

    # FP - Number of sites incorrectly detected (i.e. not correctly detected at least once)
    fp_sites = np.setdiff1d(np.intersect1d(sites_absent, sites_detected), tp_sites) # remove sites where the species was otherwise correctly detected at least once
    nsites_fp = len(fp_sites)

    # TN - Number of sites correctly not detected
    nsites_tn = len(np.intersect1d(sites_notdetected, sites_absent))

    # FN - Number of sites incorrectly not detected
    nsites_fn = len(np.intersect1d(sites_notdetected, sites_present))

    nsites_accounted_for = nsites_tp + nsites_fp + nsites_tn + nsites_fn
    if nsites_accounted_for != len(all_sites):
        print_error(f'Only {nsites_accounted_for} sites accounted for of {len(all_sites)} total')
    
    if nsites_tp + nsites_fn != len(sites_present):
        print_error(f'Incorrect true presences TP {nsites_tp} + FN {nsites_fn} != {len(sites_present)}')
    if nsites_fp + nsites_tn != len(sites_absent):
        print_error(f'Incorrect true absences FP {nsites_fp} + TN {nsites_tn} != {len(sites_absent)}')
    
    try:
        precision = nsites_tp / (nsites_tp + nsites_fp)
    except ZeroDivisionError:
        precision = np.nan
    try:
        recall = nsites_tp / (nsites_tp + nsites_fn)
    except ZeroDivisionError:
        recall = np.nan
    # try:
    #     tp_pcnt = round(nsites_tp / len(sites_present), 2)
    # except ZeroDivisionError:
    #     tp_pcnt = np.nan
    # try:
    #     tn_pcnt = round(nsites_tn / len(sites_absent), 2)
    # except ZeroDivisionError:
    #     tn_pcnt = np.nan
    
    result = {
        'label':          [species],
        'present':        [len(sites_present)],
        'absent':         [len(sites_absent)],
        'threshold':      [threshold],
        'detected':       [len(sites_detected)],
        'notdetected':    [len(sites_notdetected)],
        'correct':        [nsites_tp + nsites_tn],
        'correct_pcnt':   [round((nsites_tp + nsites_tn) / (len(sites_present) + len(sites_absent)), 2)],
        'error':          [nsites_fp + nsites_fn],
        'error_pcnt':     [round((nsites_fp + nsites_fn) / (len(sites_present) + len(sites_absent)), 2)],
        'FP': [nsites_fp],
        'FN': [nsites_fn],
        'TP': [nsites_tp],
        'TN': [nsites_tn],
        # 'TP_pcnt': [tp_pcnt],
        # 'TN_pcnt': [tn_pcnt],
        'precision': [round(precision,3)],
        'recall':    [round(recall,3)]
    }
    return(pd.DataFrame(result, index=None))