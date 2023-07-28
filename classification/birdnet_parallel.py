import pandas as pd
import glob
import os
from birdnetlib.batch import DirectoryMultiProcessingAnalyzer 
from birdnetlib.analyzer import Analyzer
import datetime

# Analyzer config
lat=47.676786
lon=-124.136721
min_conf=0.5

# File config
in_filetype = '.wav'
in_dir = "D:\\DNR\\2021\\Deployment1_2021April13_14\\S4A04325_20210414_Data"
# in_dir = 'C:\\Users\\gioj\\Desktop\\test'
out_dir = os.path.dirname(__file__) + '\\_output\\'
out_file = 'detections.csv'
out_path = out_dir + out_file

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

def on_analyze_directory_complete(recordings):
    print(f'Directory complete ({len(recordings)} recordings analyzed)')

    # Store detections in results
    results = pd.DataFrame()
    for recording in recordings:
        result = pd.DataFrame(recording.detections)
        print(f'{len(result)} detections from {recording.path}')
        info = get_info_from_filename(recording.path)
        print(f"{info['serial_no']} {info['year']}/{info['month']}/{info['day']} {info['hour']}:{info['min']}:{info['sec']}")

        if not result.empty:
            start_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['start_time'])))
            start_dates = list(map(lambda d: dt + d, start_deltas))
            result['start_date'] = start_dates

            end_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['end_time'])))
            end_dates = list(map(lambda d: dt + d, end_deltas))
            result['end_date'] = end_dates

            result['serial_no'] = info['serial_no']
            result['file'] = os.path.basename(recording.path)

            results = pd.concat([results, result], ignore_index=True)
    
    # Save to file
    pd.DataFrame.to_csv(results, out_dir + str(os.path.basename(os.path.dirname(recordings[0].path))) + '.csv', index=False)

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

if __name__ == '__main__':
    analyzer = Analyzer()

    for dir in getDirectoriesWithFiles(in_dir, in_filetype):

        # TODO: if 'dir' is in _output, we have analyzed it before, and should skip it

        print('Analyzing directory ' + dir + '...')
        info = get_info_from_filename(os.path.basename(os.path.normpath(dir))) # note 'time' here is nonsense
        print(info['serial_no'])
        print(info['date'])
        dt = datetime.datetime(int(info['year']), int(info['month']), int(info['day']))
        batch = DirectoryMultiProcessingAnalyzer(
            directory=dir,
            analyzers = [analyzer],
            date=dt,
            lat=lat,
            lon=lon,
            min_conf=min_conf,
            patterns=[('*' + in_filetype)]
        )
        batch.on_analyze_directory_complete = on_analyze_directory_complete
        print('Processing directory ' + dir +'...')
        batch.process()
        print('Done!')

    print('FINITO DIRECTERO!')

# ##########

# # Load csv of processed files

# # For each directory containing .wav files
#     # If the files in the directory haven't been processed
#         # Create a batch for the directory
#         # Process the batch
#         # Save the results