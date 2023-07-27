import pandas as pd
import glob
import os
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import datetime

# File config
in_filetype = '.wav'
out_dir = os.path.dirname(__file__) + '\\_output\\'
out_file = 'detections.csv'
out_path = out_dir + out_file

# BirdNET config 
min_conf = 0.5

# Find files to process
files = glob.glob('D:\\DNR\\2021\\**\\*' + in_filetype, recursive=True)
num_total_files = len(files)

# Ignore previously processed files, if any
if os.path.isfile(out_path):
    print('Ignoring previously processed files')
    results = pd.read_csv(out_path)
    files = [f for f in files if f not in list(set(results['file']))]

else:
    results = pd.DataFrame()

print(f'Processing files ({len(files)} of {num_total_files})...')

# Load and initialize BirdNET-Analyzer models
analyzer = Analyzer()

for file in files[0:3]:

    filename = file
    substrs = filename.split(in_filetype)[0].split('_')
    
    serial_no = substrs[0]

    date = substrs[1]
    year = date[0:4]
    month = date[4:6]
    day = date[6:8]

    time = substrs[2]
    hour = time[0:2]
    min  = time[2:4]
    sec  = time[4:6]

    print(f"{serial_no} {year}/{month}/{day} {hour}:{min}:{sec}")

    dt = datetime.datetime(int(year), int(month), int(day), int(hour), int(min), int(sec))

    # Run analyzer
    recording = Recording(
        analyzer,
        file,
        lat=47.676786,
        lon=-124.136721,
        date=dt,
        min_conf=0.5,
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

        result['serial_no'] = serial_no
        result['file'] = filename

        results = pd.concat([results, result], ignore_index=True)

print('Done! Final results:')
print(results)

pd.DataFrame.to_csv(results, out_path, index=False)
