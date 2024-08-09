import pandas as pd
import matplotlib.pyplot as plt
from annotation.annotations import *
from utils.log import *
from classification.performance import *
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

################################################################################
# Retrieve all incorrect detections above a score threhold (e.g. >= 0.9)

dirs = [ # Annotation directories to use as evaluation data
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020',
]
threshold = 0.5
species = sorted(labels.get_species_classes())

# Retrieve raw annotation data
raw_annotations = get_raw_annotations(dirs = dirs, overwrite = True)
# print(raw_annotations)

# Threshold detections
detections = raw_annotations[raw_annotations['confidence'] >= threshold].copy()

# Create a column index for unique detections
def concatenate_row(row):
    return ''.join(str(row[col]) for col in ['date', 'time', 'serialno'])
detections['detection_id'] = detections.apply(concatenate_row, axis=1)

# Remove detections with unknowns and any additional annotations for correct detections
detections = detections[~detections['detection_id'].isin(detections[detections['label_truth'] == 'unknown']['detection_id'].values)]
detections = detections[detections['label_truth'] != 'unknown']

# Determine which detections are correct
def check_correct_detection(group):
    predicted_label = group['label_predicted'].iloc[0]
    return any(group['label_truth'] == predicted_label)
correct_detections = detections.groupby('detection_id').apply(lambda group: check_correct_detection(group), include_groups=False)
detections = detections.join(correct_detections.rename('correct'), on='detection_id')

# print(detections.sort_values(by='file').to_string())

# Remove any additional annotations for correct detections
def simplify_correct_detections(group):
    if group['correct'].all():  # Check if all rows in the group have 'correct' == True
        return group[group['label_truth'] == group['label_predicted'].iloc[0]]
    else:
        return group
detections = detections.reset_index(drop=True).groupby('detection_id').apply(lambda g: simplify_correct_detections(g), include_groups=False)
detections['detection_id'] = detections.apply(concatenate_row, axis=1)

# Remove any duplicate labels for each detection
def remove_duplicates(group):
    return group.drop_duplicates(subset=['label_predicted', 'label_truth'], keep='first')
detections = detections.reset_index(drop=True).groupby('detection_id').apply(lambda g: remove_duplicates(g), include_groups=False)
detections = detections.sort_values(by='file').reset_index(drop=True)

# TODO: Create a "biotic_concurrent" entry for all incorrect detections
# TODO

# print(detections.to_string())

# Print label incorrect counts
incorrect_detections = detections[detections['correct'] == False]
print_warning(incorrect_detections[incorrect_detections['confidence'] < 0.9].reset_index(drop=True).to_string())
incorrect_detection_counts = incorrect_detections['label_truth'].value_counts()
print(incorrect_detection_counts)

# Remove any "not_target" labels
detections = detections[detections['label_truth'] != 'not_target']

################################################################################
# Construct a confusion matrix

class_labels = sorted(pd.unique(detections[['label_predicted', 'label_truth']].values.ravel()))
print(class_labels)
confusion_mtx = confusion_matrix(detections["label_truth"], detections["label_predicted"], labels=class_labels)

df_confusion_mtx = pd.DataFrame(confusion_mtx, class_labels, class_labels)

print(confusion_mtx)
print(confusion_mtx.shape)
print(len(class_labels))
print(df_confusion_mtx)

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=class_labels)
# disp.plot()
# plt.show()

import seaborn as sn
sn.set_theme(font_scale=0.5)
cm = sn.heatmap(confusion_mtx, annot=True, vmin=0, vmax=5, cmap="Reds", xticklabels=class_labels, yticklabels=class_labels)
cm.set_xlabel("Predicted label")
cm.set_ylabel("True label")
plt.show()

################################################################################
# Export data for hierarchical edge bundling plot in R

confusion_matrix_filepath = 'data/annotations/processed/confusion_matrix.csv'
df_confusion_mtx.to_csv(confusion_matrix_filepath, index=True)
print_success(f'Saved confusion matrix to {confusion_matrix_filepath}')

################################################################################
# Identify labels for improvement
# - a) the label is of high relevance for management or conservation objectives
# - b) pre-trained model performance for the label is insufficient
# - c) the label is a recurrent source of confusion for other labels or a novel and commonly encountered signal in the target domain
# TODO
