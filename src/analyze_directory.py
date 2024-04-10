# Analyze directories of files in parallel

# NOTE: Custom edits to birdnetlib required (see analyzer.py and main.py).
# Specifically, APPLY_SIGMOID flag set to False throughout to
# return logits, not sigmoid activations

from classification import process_dir
import os

# Input and output config
in_dir      = r'/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020' # e.g. r'/Volumes/gioj_b1/OESF/2020/Deployment8'
root_dir    = r'/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020' # retain directory structure relative to this root, e.g. r'/Volumes/gioj_b1/OESF'
out_dir     = r'/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/sound_separation_4_detections' # e.g. r'/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/raw_detections'
in_filetype = '.wav'
sort_by     = 'start_date' # (e.g. start_date, confidence)
ascending   = True

# Analyzer config
n_processes    = 7 # cores in pool
min_confidence = 0.0
num_separation = 1
cleanup        = True

# -----------------------------------------------------------------------------

# Normalize file paths to support both mac and windows
in_dir   = os.path.normpath(in_dir)
root_dir = os.path.normpath(root_dir)
out_dir  = os.path.normpath(out_dir)

if __name__ == '__main__':
    process_dir.process_dir_parallel(
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
