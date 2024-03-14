##
# Validate evaluation data and evaluate classifier performance for each species class
# Requires annotations from Google Drive. Ensure these are downloaded locally before running.

# dirs = [
#     'TODO'
# ]

dirs = [
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment4/SMA00410_20200523', # Jack
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment4/SMA00424_20200521', # Stevan
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment4/SMA00486_20200523', # Summer
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment4/SMA00556_20200524', # Gio
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment6/SMA00556_20200618', # Stevan
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment6/SMA00349_20200619', # Mirella
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment6/SMA00399_20200619', # Jessica
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment8/SMA00346_20200716', # Summer
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020/Deployment8/SMA00339_20200717'  # Jack
]

print_annotations = False

# Load required packages -------------------------------------------------
from annotation.annotations import *

# Collate and validate annotation data
raw_annotations = collate_annotations(dirs, print_annotations=False)
