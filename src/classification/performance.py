# Evaluate performance for a single class (species)

import pandas as pd             # Data manipulation
import matplotlib.pyplot as plt # Plotting
import sklearn.metrics          # Classifier evaluation
import numpy as np              # Mathematics
from utils.log import *

# detection_labels - a dataframe of detections with columns 'label_truth' (where a positive presence is represented by the species class label) and 'confidence'
# species - the species class label, e.g. "american crow"
def evaluate_species_performance(detection_labels, species, plot):
    
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

    padding = 0.01
    font_size = 9

    if plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        # Plot precision and recall as a function of threshold
        ax1.plot(thresholds, precision[:-1], label='Precision') #, marker='.' 
        ax1.plot(thresholds, recall[:-1], label='Recall') # , marker='.'
        ax1.set_xlabel('Threshold') 
        ax1.set_ylabel('Performance')
        ax1.set_title(f'{species}\nThreshold performance', fontsize=font_size)
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
        ax2.set_title(f'{species}\nPrecision-Recall (AUC {pr_auc:.2f}, AP {pr_ap:.2f})', fontsize=font_size)
        ax2.set_xlim([0.0-padding, 1.0+padding])
        ax2.set_ylim([0.0-padding, 1.0+padding])
        ax2.legend(loc='lower left')
        ax2.set_box_aspect(1)

    # TODO: Use F1-score to determine an optimal threshold, e.g. https://ploomber.io/blog/threshold/

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
    
    # Return the performance metrics
    return pd.DataFrame({
        'species':   [species],
        'AUC-PR':    [round(pr_auc, 2)],                                # Precision-Recall AUC
        'AP':        [round(pr_ap, 2)],                                 # Average precision
        'p_mean':    [round(precision.mean(),2)],                       # Average precision across all thresholds
        'p_max':     [round(precision[np.argmax(precision[:-1])],2)],   # Maximum precision across all thresholds
        'p_max_th':  [round(thresholds[np.argmax(precision[:-1])],2)],  # Score threshold to maximize precision
        'p_max_r':   [round(recall[np.argmax(precision[:-1])],2)],      # Recall at maximum precision using threshold
        'N':         [n_examples],                                      # Total number of examples (not including "unknown" examples)
        'N_P':       [n_P],                                             # Total number of positive examples
        'N_N':       [n_N],                                             # Total number of negative examples
        'N_unknown': [n_unknown]                                        # Total number of unknown examples excluded from evaluation
    })
