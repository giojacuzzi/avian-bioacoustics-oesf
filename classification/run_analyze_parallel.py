from multiprocessing import Pool
from tools import *
from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording 
import datetime
import os
import pandas as pd
import time
import analyze
from itertools import repeat

# File config
in_filetype = '.wav'

# FOR PROCESSING RAW AUDIO FROM ENTIRE DEPLOYMENTS ---
# in_dir = '/Volumes/gioj/OESF/2020/Deployment5/S4A04325_20200603'
# root_dir = '/Volumes/gioj/OESF'
# out_dir = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/raw_detections'
# ---

# FOR PROCESSING AUDIO SUBSAMPLES FOR ANNOTATION REFERENCE ---
# root_dir = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/SMA00351_20200412'
# in_dir = root_dir
# out_dir = in_dir + '/predictions'
# ---

## FOR TESTING
root_dir = '/Users/giojacuzzi/Desktop/audio_test/1'
in_dir = root_dir
out_dir = in_dir + '/predictions'
# sort_by = 'start_time' # 'confidence' # TODO
## /TESTING

# Analyzer config
n_processes = 8 # cores per batch
min_confidence = 0.0
num_separation = 4
cleanup = True
sort_by = 'confidence'
ascending = False
species_list_path = os.path.abspath('classification/species_list/species_list_OESF.txt')

# -----------------------------------------------------------------------------

if 'analyzer' not in locals() and 'analyzer' not in globals():
    analyzer = Analyzer(custom_species_list_path=species_list_path)

def process_file(filepath, sort_by = 'start_date', ascending = False):
    file_out = os.path.splitext(filepath[len(root_dir):])[0] + '.csv'
    path_out = out_dir + file_out

    already_analyzed = list_base_files_by_extension(out_dir, 'csv')
    already_analyzed = [f.rstrip('.csv') for f in already_analyzed]
    if (os.path.splitext(os.path.basename(filepath))[0]) in already_analyzed:
        print(f'  {os.path.basename(filepath)} already analyzed. SKIPPING...')
        return

    # Run analyzer to obtain detections
    try:
        start_time_file = time.time()
        result = analyze.analyze_detections(
            filepath = filepath,
            analyzer = analyzer,
            min_confidence = min_confidence,
            num_separation = num_separation,
            cleanup = cleanup
        )

        info = get_info_from_filename(filepath)
        dt = datetime.datetime(int(info['year']), int(info['month']), int(info['day']), int(info['hour']), int(info['min']), int(info['sec']))

        col_names = ['common_name','confidence','start_date']
        if not result.empty:
            start_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['start_time'])))
            start_dates = list(map(lambda d: dt + d, start_deltas))
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

if __name__ == '__main__':
    dirs = getDirectoriesWithFiles(in_dir, in_filetype)
    dirs.sort()

    for dir in dirs:
        print('Processing directory ' + dir +'...')
        start_time_dir = time.time()
        files = list_files_in_directory(dir)
        files = [f for f in files if f.endswith(in_filetype)]
        files.sort()
        n_processes = min(len(files), n_processes)
        with Pool(n_processes) as pool: # start batch pool for all files in directory
            # pool.map(process_file, files, sort_by = 'confidence')
            pool.starmap(process_file, zip(files, repeat(sort_by), repeat(ascending)))
        end_time_dir = time.time()
        print(f'Finished directory {dir}\n({end_time_dir - start_time_dir} sec). Proceeding to next...')

    print(f'Finished analyzing all directories in {in_dir}!')
