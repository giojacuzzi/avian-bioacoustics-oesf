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
from process_file import process_file

# Run the analyzer on a directory of files in parallel, creating a csv for each file
def process_dir_parallel(
        in_dir,
        out_dir,
        in_filetype,
        root_dir=None,
        n_processes=8, # cores per batch
        min_confidence=0.0,
        num_separation=1,
        cleanup = True,
        sort_by='start_date',
        ascending=True
):
    if root_dir is None:
        root_dir = in_dir
    dirs = getDirectoriesWithFiles(in_dir, in_filetype)
    dirs.sort()

    for dir in dirs:
        print('Processing directory ' + dir +'...')

        # DEBUG - skip sm2
        if 'SM2' in os.path.basename(dir):
            print('SM2 directory, skipping!')
            continue

        start_time_dir = time.time()
        files = list_files_in_directory(dir)
        files = [f for f in files if f.endswith(in_filetype)]
        files.sort()
        with Pool(min(len(files), n_processes)) as pool: # start batch pool for all files in directory
            # pool.map(process_file, files, sort_by = 'confidence')
            pool.starmap(process_file, zip(
                files,
                repeat(out_dir),
                repeat(min_confidence),
                repeat(num_separation),
                repeat(cleanup),
                repeat(root_dir),
                repeat(sort_by),
                repeat(ascending)
            ))
        end_time_dir = time.time()
        print(f'Finished directory {dir}\n({end_time_dir - start_time_dir} sec). Proceeding to next...')
