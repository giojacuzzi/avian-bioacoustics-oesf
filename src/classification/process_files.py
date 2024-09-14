# NOTE: Custom edits to birdnetlib required (see analyzer.py).
# Specifically, APPLY_SIGMOID flag set to False throughout to
# return logits, not sigmoid activations

from multiprocessing import Pool
import os
import time
from itertools import repeat
from .process_file import process_file
from utils.log import *
from utils.files import *

# Run the analyzer on a directory of files in parallel, creating a csv for each file
def process_files_parallel(
        in_files,
        out_dir,
        root_dirs      = [],
        in_filetype    = '.wav',
        analyzer_filepath = None,
        labels_filepath = 'src/classification/species_list/species_list_OESF.txt',
        n_processes    = 8, # cores per batch
        min_confidence = 0.0,
        apply_sigmoid  = True,
        num_separation = 1,
        cleanup        = True,
        sort_by        = 'start_date',
        ascending      = True,
        save_to_file   = True
):
    print('process_files_parallel')
    if len(root_dirs) != 0 and len(root_dirs) != len(in_files):
        print_error('Number of root directories must match input directories')
        return

    start_time_dir = time.time()
    with Pool(min(len(in_files), n_processes)) as pool: # start process pool for all files in directory
        pool.starmap(process_file, zip(
            in_files,                  # in_filepath
            repeat(out_dir),        # out_dir
            root_dirs,       # root_dir
            repeat(analyzer_filepath),
            repeat(labels_filepath),
            repeat(min_confidence), # min_confidence
            repeat(apply_sigmoid),  # apply_sigmoid
            repeat(num_separation), # num_separation
            repeat(cleanup),        # cleanup
            repeat(sort_by),        # sort_by
            repeat(ascending),      # ascending
            repeat(save_to_file)    # save_to_file
        ))
    end_time_dir = time.time()
    print(f'Finished processing files\n({end_time_dir - start_time_dir} sec)')


# Run the analyzer on a directory of files in parallel, creating a csv for each file
def process_dir_parallel(
        in_dir,
        out_dir,
        root_dir       = None,
        in_filetype    = '.wav',
        n_processes    = 8, # cores per batch
        min_confidence = 0.0,
        apply_sigmoid  = True,
        num_separation = 1,
        cleanup        = True,
        sort_by        = 'start_date',
        ascending      = True,
        save_to_file   = True
):
    
    if root_dir is not None and not root_dir in in_dir:
        print_error('Root directory must contain input directory')
        return

    print('getting dirs...')
    dirs = getDirectoriesWithFiles(in_dir, in_filetype)
    dirs.sort()

    print('printing dirs...')
    print(dirs)

    for dir in dirs:
        print('Processing directory ' + dir +'...')

        # NOTE: skip model SM2 recordings
        if 'SM2' in os.path.basename(dir):
            print_warning('Skipping SM2 directory...')
            continue

        start_time_dir = time.time()
        files = list_files_in_directory(dir)
        files = [f for f in files]
        files.sort()
        print(files)
        print(f'LAUNCH {n_processes}')
        with Pool(min(len(files), n_processes)) as pool: # start process pool for all files in directory
            pool.starmap(process_file, zip(
                files,                  # in_filepath
                repeat(out_dir),        # out_dir
                repeat(root_dir),       # root_dir
                repeat(None),           # analyzer_filepath
                repeat('src/classification/species_list/species_list_OESF.txt'), # labels_filepath
                repeat(min_confidence), # min_confidence
                repeat(apply_sigmoid),  # apply_sigmoid
                repeat(num_separation), # num_separation
                repeat(cleanup),        # cleanup
                repeat(sort_by),        # sort_by
                repeat(ascending),      # ascending
                repeat(save_to_file)    # save_to_file
            ))
        end_time_dir = time.time()
        print(f'Finished directory {dir}\n({end_time_dir - start_time_dir} sec). Proceeding to next...')
