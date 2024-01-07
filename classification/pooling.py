from multiprocessing import Pool
from helper import *
from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording 
import datetime
import os
import pandas as pd
import time

# Batching config
n_processes = 8

# Analyzer config
lat=47.676786
lon=-124.136721

if 'analyzer' not in locals() and 'analyzer' not in globals():
    analyzer = Analyzer()

# File config
in_filetype = '.wav'
in_dir = '/Volumes/gioj_b1/OESF/2020/Deployment4'
root_dir = '/Volumes/gioj_b1/OESF'
out_dir = os.path.dirname(__file__) + '/_output'

def analyze_file(file):
    file_out = os.path.splitext(file[len(root_dir):])[0] + '.csv'
    path_out = out_dir + file_out

    already_analyzed = list_base_files_by_extension(out_dir, 'csv')
    already_analyzed = [f.rstrip('.csv') for f in already_analyzed]
    if (os.path.splitext(os.path.basename(file))[0]) in already_analyzed:
        print(f'  {os.path.basename(file)} already analyzed. SKIPPING...')
        return

    # TODO: move into try-catch
    # print(f'  Analyzing {os.path.basename(file)}...')
    info = get_info_from_filename(file)
    start_time_file = time.time()
    dt = datetime.datetime(int(info['year']), int(info['month']), int(info['day']), int(info['hour']), int(info['min']), int(info['sec']))

    try:
        # Run analyzer
        recording = Recording(
            analyzer=analyzer, path=file,
            lat=lat, lon=lon, date=dt,
            # min_conf=0.1,
        )
        recording.analyze()

        # Store detections in results
        result = pd.DataFrame(recording.detections)
        print(str(len(result)) + ' detections')

        col_names = ['common_name','confidence','start_date','end_date']
        if not result.empty:
            start_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['start_time'])))
            start_dates = list(map(lambda d: dt + d, start_deltas))
            result['start_date'] = start_dates

            end_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['end_time'])))
            end_dates = list(map(lambda d: dt + d, end_deltas))
            result['end_date'] = end_dates

            result = result[col_names] # only keep essential values
        else:
            result = pd.DataFrame(columns=col_names)

        if not os.path.exists(os.path.dirname(path_out)):
            os.makedirs(os.path.dirname(path_out))
        pd.DataFrame.to_csv(result, path_out, index=False) 

        end_time_file = time.time()
        print(f'Finished file {file}\n({end_time_file - start_time_file} sec)')

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
        with Pool(n_processes) as p: # start batch pool for all files in directory
            p.map(analyze_file, files)
        end_time_dir = time.time()
        print(f'Finished directory {dir}\n({end_time_dir - start_time_dir} sec). Proceeding to next...')

    print(f'Finished analyzing all directories in {in_dir}!')
