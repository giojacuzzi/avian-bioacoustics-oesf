# Analyze directories of files in parallel

# NOTE: Custom edits to birdnetlib required (see analyzer.py).
# Specifically, APPLY_SIGMOID flag set to False throughout to
# return logits, not sigmoid activations
from birdnetlib.analyzer import Analyzer

from multiprocessing import Pool
from tools import *
import datetime
import os
import pandas as pd
import time
import analyze
from itertools import repeat
from process_dir import process_dir_parallel

# FOR PROCESSING RAW AUDIO FROM ENTIRE DEPLOYMENTS ---
in_dir = '/Volumes/gioj_b1/OESF/2020/Deployment8'
root_dir = '/Volumes/gioj_b1/OESF'
out_dir = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/raw_detections'
sort_by='start_date'
ascending=True
# ---

## FOR TESTING
# root_dir = '/Users/giojacuzzi/Desktop/audio_test/full'
# in_dir = root_dir
# out_dir = in_dir + '/predictions'
# sort_by = 'confidence'
# ascending = False
## /TESTING

# Analyzer config
n_processes = 7 # cores per batch
min_confidence = 0.0
num_separation = 1
cleanup = False

# File config
in_filetype = '.wav'

# -----------------------------------------------------------------------------

# RUN
if __name__ == '__main__':
    out_dir = out_dir + in_dir.replace(root_dir, '') # preserve the directory structure of the original data
    process_dir_parallel(
        in_dir = in_dir,
        out_dir = out_dir,
        in_filetype=in_filetype,
        n_processes=n_processes,
        min_confidence=min_confidence,
        num_separation=num_separation,
        cleanup=cleanup,
        sort_by=sort_by,
        ascending=ascending
    )
    print(f'Finished analyzing all directories in {in_dir}!')
