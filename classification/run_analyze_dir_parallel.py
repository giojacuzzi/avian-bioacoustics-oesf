# Analyze directories of files in parallel

# NOTE: Custom edits to birdnetlib required (see analyzer.py).
# Specifically, APPLY_SIGMOID flag set to False throughout to
# return logits, not sigmoid activations

from tools import *
from process_dir import process_dir_parallel

# Input and output config
in_dir    = '/Volumes/gioj_b1/OESF/2020/Deployment8/SMA00339_20200717'
root_dir  = '/Volumes/gioj_b1/OESF' # retain directory structure relative to this root
out_dir   = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/raw_detections'
sort_by   = 'start_date' # (e.g. start_date, confidence)
ascending = True

# Analyzer config
n_processes = 7 # cores in pool
min_confidence = 0.0
num_separation = 1
cleanup = False

# File config
in_filetype = '.wav'

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    process_dir_parallel(
        in_dir         = in_dir,
        out_dir        = out_dir,
        root_dir       = root_dir,
        in_filetype    = in_filetype,
        n_processes    = n_processes,
        min_confidence = min_confidence,
        num_separation = num_separation,
        cleanup        = cleanup,
        sort_by        = sort_by,
        ascending      = ascending
    )
    print(f'Finished analyzing all directories in {in_dir}!')
