# # For each directory containing .wav files
#     # If the files in the directory haven't been processed
#         # Create a batch for the directory
#         # Process the batch
#         # Save the results

from birdnetlib.batch import DirectoryMultiProcessingAnalyzer
from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording 
import datetime
import os
import pandas as pd
import time

# Analyzer config
lat=47.676786
lon=-124.136721
processes=4

# File config
in_filetype = '.wav'
in_dir = '/Volumes/gioj_b1/OESF'
# in_dir = '/Volumes/gioj_b1/OESF/2020/Deployment2'
# in_dir = '/Users/giojacuzzi/Desktop/audio_test/parallel_test'
out_dir = os.path.dirname(__file__) + '/_output'

# Scrape serial number, date, and time from Song Meter filename
def get_info_from_filename(path):
    filename = os.path.basename(path)
    # print(filename)
    substrs = filename.split(in_filetype)[0].split('_')
    # print(substrs)
    date = substrs[1]
    # print(len(substrs))
    if len(substrs) > 2:
        time = substrs[2]
    else:
        time = '000000' # default time for directories rather than individual files
    return({
        'serial_no': substrs[0],
        'date':      date,
        'year':      date[0:4],
        'month':     date[4:6],
        'day':       date[6:8],
        'time':      time,
        'hour':      time[0:2],
        'min':       time[2:4],
        'sec':       time[4:6],
    })

def getDirectoriesWithFiles(path, filetype):
    directoryList = []
    if os.path.isfile(path):
        return []
    # Add dir to directorylist if it contains files of filetype
    if len([f for f in os.listdir(path) if f.endswith(filetype)]) > 0:
        directoryList.append(path)
    for d in os.listdir(path):
        new_path = os.path.join(path, d)
        if os.path.isdir(new_path):
            directoryList += getDirectoriesWithFiles(new_path, filetype)
    return directoryList

def list_base_files_by_extension(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(f'.{extension}'):
                file_list.append(filename)
    return file_list

def on_analyze_directory_complete(recordings):
    print("-" * 80)
    print("directory_completed: recordings processed ", len(recordings))
    print("-" * 80)
    results = pd.DataFrame()

    for recording in recordings:
        print(recording.path)
        if recording.error:
            print("Error with recording: ", recording.error_message)
        else:
            # Store detections in results
            result = pd.DataFrame(recording.detections)
            print(str(len(result)) + ' detections')

            if not result.empty:
                start_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['start_time'])))
                start_dates = list(map(lambda d: dt + d, start_deltas))
                result['start_date'] = start_dates

                end_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['end_time'])))
                end_dates = list(map(lambda d: dt + d, end_deltas))
                result['end_date'] = end_dates

                file_out = os.path.splitext(recording.path[len(in_dir):])[0] + '.csv'
                path_out = out_dir + file_out
                result = result[['common_name','confidence','start_date','end_date']] # only keep essential values

                if not os.path.exists(os.path.dirname(path_out)):
                    os.makedirs(os.path.dirname(path_out))
                pd.DataFrame.to_csv(result, path_out, index=False) 

                results = pd.concat([results, result], ignore_index=True)

if __name__ == '__main__':
# if True:
    analyzer = Analyzer()

    dirs = getDirectoriesWithFiles(in_dir, in_filetype)
    dirs.sort()

    for dir in dirs:

        # check if directory has already been analyzed, not file
        already_analyzed = []
        for r, dd, ff in os.walk(out_dir):
            for d in dd:
                already_analyzed.append(d)

        if os.path.basename(dir) in already_analyzed:
            print('Directory ' + os.path.basename(dir) +' already analyzed. Skipping...')
            continue
        print('Processing directory ' + os.path.basename(dir) +'...')
        start_time_dir = time.time()

        info = get_info_from_filename(dir)
        dt = datetime.datetime(int(info['year']), int(info['month']), int(info['day']))

        try:
            batch = DirectoryMultiProcessingAnalyzer(
                dir,
                analyzers=[analyzer],
                lon=lon, lat=lat,
                date=dt, # this should be removed in the official runthrough, and use species list instead
                # min_conf=0.1,
                processes=processes
            )

            batch.on_analyze_directory_complete = on_analyze_directory_complete
            print('Starting batch process...')
            batch.process()
        except Exception as e:
            print(f'EXCEPTION: {str(e)}')

        end_time_dir = time.time()
        print(f'Finished batch process for {os.path.basename(dir)}({end_time_dir - start_time_dir} sec). Proceeding to next directory...')

    print(f'Finished analyzing all directories in {in_dir}')
