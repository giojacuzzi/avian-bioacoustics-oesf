# # For each directory containing .wav files
#     # If the files in the directory haven't been processed
#         # Create a batch for the directory
#         # Process the batch
#         # Save the results

from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording 
import datetime
import os
import pandas as pd
import time

# Analyzer config
lat=47.676786
lon=-124.136721

# File config
in_filetype = '.wav'
in_dir = '/Volumes/gioj_b1/OESF'
in_dir = '/Volumes/gioj_b1/OESF/2020/Deployment4'
out_dir = os.path.dirname(__file__) + '/_output'

# Scrape serial number, date, and time from Song Meter filename
def get_info_from_filename(path):
    filename = os.path.basename(path)
    substrs = filename.split(in_filetype)[0].split('_')
    date = substrs[1]
    time = substrs[2]
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

if __name__ == '__main__':
    analyzer = Analyzer()

    dirs = getDirectoriesWithFiles(in_dir, in_filetype)
    dirs.sort()

    already_analyzed = list_base_files_by_extension(out_dir, 'csv')
    already_analyzed = [f.rstrip('.csv') for f in already_analyzed]

    for dir in dirs:
        print('Processing directory ' + dir +'...')
        start_time_dir = time.time()

        files = os.listdir(dir)
        files = [f for f in files if f.endswith(in_filetype)]
        files.sort()

        results = pd.DataFrame()
        for file in files:
            file = os.path.join(dir, file)
            file_out = os.path.splitext(file[len(in_dir):])[0] + '.csv'
            path_out = out_dir + file_out

            if (os.path.splitext(os.path.basename(file))[0]) in already_analyzed:
                print(f'  {os.path.basename(file)} already analyzed. SKIPPING...')
                continue

            print(f'  Processing {os.path.basename(file)}...')
            info = get_info_from_filename(file)
            start_time_file = time.time()
            # print(f"Processing {info['serial_no']} {info['year']}/{info['month']}/{info['day']} {info['hour']}:{info['min']}:{info['sec']}")

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

                if not result.empty:
                    start_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['start_time'])))
                    start_dates = list(map(lambda d: dt + d, start_deltas))
                    result['start_date'] = start_dates

                    end_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['end_time'])))
                    end_dates = list(map(lambda d: dt + d, end_deltas))
                    result['end_date'] = end_dates

                    # result['serial_no'] = info['serial_no']
                    result['file'] = file
                    result = result[['common_name','confidence','start_date','end_date','file']] # only keep essential values

                    if not os.path.exists(os.path.dirname(path_out)):
                        os.makedirs(os.path.dirname(path_out))
                    pd.DataFrame.to_csv(result.sort_values(by='file'), path_out, index=False) 

                    results = pd.concat([results, result], ignore_index=True)

                end_time_file = time.time()
                print(f'Finished file {file}\n({end_time_file - start_time_file} sec)')

            except Exception as e:
                print(f'EXCEPTION: {str(e)}')

        end_time_dir = time.time()
        print(f'Finished directory {dir}\n({end_time_dir - start_time_dir} sec) Moving on to the next directory...')

    print(f'Finished analyzing all directories in {in_dir}!')
