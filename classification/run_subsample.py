# Split a specified site-day into 1-minute subsamples

# ENTER THESE VALUES MANUALLY
id    = 'SMA00351'
lat   = 47.67369
lon   = -124.33503
year  = 2020
month_start = 4
month_end   = 4
day_start   = 12
day_end     = 21
path  = f'/Volumes/gioj/OESF/2020/Deployment1/SMA00351_20200412_Data'
#
outpath = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator'

from subsample import *
import numpy as np
from pydub import AudioSegment
import tools
from run_analyze_parallel import process_file

# Create output directory
output = outpath + f'/{id}_{year}{month_start:02d}{day_start:02d}'
if not os.path.exists(output):
    os.makedirs(output)

# Calculate subsample times
sample_times = get_subsample_datetimes(id, lat, lon, year, month_start, month_end, day_start, day_end)
# for s in sample_times: print(s)

# Create subsample wavs for each date
for date in pd.date_range(start=datetime(year=year, month=month_start, day=day_start), end=datetime(year=year, month=month_end, day=day_end)):
    print(f'Subsampling date {date.strftime("%Y-%m-%d")}...')
    datetimes_with_same_date = [time for time in sample_times if time.date() == date.date()]
    # for dt in datetimes_with_same_date:
    #     print(dt.strftime("%Y-%m-%d %H:%M:%S"))

    # Combine recordings from day into single data structure
    files = find_files_by_date(path, date.year, date.month, date.day)
    data = AudioSegment.empty()
    print('Combining recordings...')
    for f in files:
        print(f' {f}')
        metadata = get_metadata_from_filename(f)
        hour_start = int(metadata['hour'])
        sec_start = int(metadata['second'])

        w = AudioSegment.from_wav(f)
        w = tools.remove_dc_offset(w)

        data_hr = AudioSegment.silent(duration=3600 * 1000, frame_rate=w.frame_rate)
        data_hr = data_hr.overlay(w, position=sec_start * 1000)
        data = data + data_hr

    # Save N minute subsamples to file
    subsample_len = 12 # seconds per subsample
    print(f'Saving {subsample_len} second subsamples to file...')
    for t in datetimes_with_same_date:
        t_outpath = f'{output}/{id}_{t.year}{t.month:02d}{t.day:02d}_{t.hour:02d}{t.minute:02d}{t.second:02d}.wav'
        print(t_outpath)
        start_time_ms = (t.hour * 60 * 60 + t.minute * 60 + t.second) * 1000
        end_time_ms = start_time_ms + subsample_len * 1000 # N minute subsample
        data_subsample = data[start_time_ms:end_time_ms]
        data_subsample.export(t_outpath, format='wav')

        # use classifier with sound separation to make predictions for reference
        # TODO: run classifier on the data
        # TODO: save to a folder as a .csv file
