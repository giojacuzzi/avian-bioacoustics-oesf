# Analyze directories of files in parallel

from classification import process_files
import os
import shutil
from utils.log import *

# Input and output config
in_dir      = '/Volumes/gioj_w2/OESF/2020/'
root_dir    = '/Volumes/gioj_w2/OESF' # Directory structure below root_dir will be retrained in out_dir
out_dir     = '/Users/giojacuzzi/Downloads/CUSTOM_RESULTS'
in_filetype = '.wav'
sort_by     = 'start_date' # (e.g. start_date, confidence)
ascending   = True
overwrite   = False

# Analyzer config
use_custom_model = True # If False, will use pre-trained model
n_processes      = 7      # Number of cores used by the processing pool. Recommended <= number of physical cores available on your computer.
min_confidence   = 0.0
apply_sigmoid    = True
num_separation   = 1
cleanup          = True

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    print(f'Beginning analysis of all {in_filetype} files at {in_dir}...')

    if use_custom_model: # Target (custom) model
        analyzer_filepath = 'data/models/custom/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0.tflite'
        labels_filepath   = 'data/models/custom/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0_Labels.txt'
    else: # Source (pre-trained) model
        analyzer_filepath = None
        labels_filepath   = 'src/classification/species_list/species_list_OESF.txt'

    # Normalize file paths to support both mac and windows
    in_dir   = os.path.normpath(in_dir)
    root_dir = os.path.normpath(root_dir)
    out_dir  = os.path.normpath(out_dir)

    # Overwrite existing output directory if requested
    if overwrite and os.path.isdir(out_dir):
        print_warning(f'Overwriting output directory {out_dir}')
        shutil.rmtree(out_dir)

    process_files.process_dir_parallel(
        in_dir            = in_dir,
        out_dir           = out_dir,
        root_dir          = root_dir,
        in_filetype       = in_filetype,
        analyzer_filepath = analyzer_filepath,
        labels_filepath   = labels_filepath,
        n_processes       = n_processes,
        min_confidence    = min_confidence,
        apply_sigmoid     = apply_sigmoid,
        num_separation    = num_separation,
        cleanup           = cleanup,
        sort_by           = sort_by,
        ascending         = ascending
    )
    print_success(f'Finished analyzing all directories in {in_dir}. Results stored at {out_dir}')
