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

# FOR PROCESSING RAW AUDIO FROM ENTIRE DEPLOYMENTS ---
in_dir = '/Volumes/gioj_b1/OESF/2020/Deployment6/SMA00556_20200618'
root_dir = '/Volumes/gioj_b1/OESF'
out_dir = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/raw_detections'
sort_by='start_date'
ascending=True
# ---

# ## FOR TESTING
# root_dir = '/Users/giojacuzzi/Desktop/audio_test/chorus'
# in_dir = root_dir
# out_dir = in_dir + '/predictions'
# sort_by = 'confidence'
# ascending = False
# ## /TESTING

# Analyzer config
n_processes = 7 # cores per batch
min_confidence = 0.0
num_separation = 1
cleanup = False

# File config
in_filetype = '.wav'

# -----------------------------------------------------------------------------

# Create a global analyzer instance
if 'analyzer' not in locals() and 'analyzer' not in globals():
    analyzer = Analyzer(custom_species_list_path=os.path.abspath('classification/species_list/species_list_OESF.txt'))

# Run the analyzer on the given file and save the resulting detections to a csv
def process_file(
        filepath,
        min_confidence=0.0,
        num_separation=1,
        cleanup=True,
        root_dir=None,
        sort_by='start_date',
        ascending=True
):
    file_out = os.path.splitext(filepath[len(root_dir):])[0] + '.csv'
    path_out = out_dir + file_out

    already_analyzed = list_base_files_by_extension(out_dir, 'csv')
    already_analyzed = [f.rstrip('.csv') for f in already_analyzed]
    if (os.path.splitext(os.path.basename(filepath))[0]) in already_analyzed:
        print(f'  {os.path.basename(filepath)} already analyzed. SKIPPING...')
        return

    # Run analyzer to obtain detections
    try:
        info = parse_metadata_from_filename(filepath)

        start_time_file = time.time()
        result = analyze.analyze_detections(
            filepath = filepath,
            analyzer = analyzer,
            min_confidence = min_confidence,
            num_separation = num_separation,
            cleanup = cleanup
        )

        if info is None:
            print('None!')
            dt = datetime.timedelta(0)
        else:
            dt = datetime.datetime(int(info['year']), int(info['month']), int(info['day']), int(info['hour']), int(info['min']), int(info['sec']))

        col_names = ['common_name','confidence','logit','start_date']
        if not result.empty:
            start_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['start_time'])))
            start_dates = list(map(lambda d: dt + d, start_deltas))

            # Create columns for raw logit value, rounded sigmoid activated confidence, and start date
            result = result.rename(columns={'confidence': 'logit'})
            result['confidence'] = sigmoid_BirdNET(result['logit'])
            result['start_date'] = start_dates

            # end_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['end_time'])))
            # end_dates = list(map(lambda d: dt + d, end_deltas))
            # result['end_date'] = end_dates

            result = result[col_names] # only keep essential values
        else:
            result = pd.DataFrame(columns=col_names)

        # sort the results
        result = result.sort_values(sort_by, ascending=ascending)

        if not os.path.exists(os.path.dirname(path_out)):
            os.makedirs(os.path.dirname(path_out))
        pd.DataFrame.to_csv(result, path_out, index=False) 

        end_time_file = time.time()
        print(f'Finished file {filepath}\n({end_time_file - start_time_file} sec)')

    except Exception as e:
        print(f'EXCEPTION: {str(e)}')

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
                repeat(min_confidence),
                repeat(num_separation),
                repeat(cleanup),
                repeat(root_dir),
                repeat(sort_by),
                repeat(ascending)
            ))
        end_time_dir = time.time()
        print(f'Finished directory {dir}\n({end_time_dir - start_time_dir} sec). Proceeding to next...')

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
