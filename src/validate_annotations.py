##
# Validate evaluation data and evaluate classifier performance for each species class
# Requires annotations from Google Drive. Ensure these are downloaded locally before running.

dirs = [
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020',
]

# Load required packages -------------------------------------------------
from annotation.annotations import *

# Retrieve and validate annotation data
raw_annotations = get_raw_annotations(dirs, overwrite=True, print_annotations=False)
