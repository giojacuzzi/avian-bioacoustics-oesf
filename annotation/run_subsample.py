# Split a specified site-day into 1-minute subsamples

# ENTER THESE VALUES MANUALLY
id    = 'SMA00404'
lat   = 47.64477
lon   = -124.19109
year  = 2023
month = 5
day   = 18
path  = f'/Volumes/gioj/OESF/2023/D1_20230518_20230528/SMA00404_20230518_000000'
#

from subsample import *
import numpy as np
from pydub import AudioSegment
import helpers

# Create output directory
outpath = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/Annotation/Data'
output = outpath + f'/{id}_{year}{month:02d}{day:02d}'
if not os.path.exists(output):
    os.makedirs(output)

# Merge audio for entire day
path_data = path + '/Data'
files = find_files_by_date(path_data, year, month, day)

# Calculate subsample times
sample_times = subsample(id, lat, lon, year, month, day)

# Combine recordings from day into single data structure
data = AudioSegment.empty()
for f in files:
    print(f)
    metadata = get_metadata_from_filename(f)
    hour_start = int(metadata['hour'])
    sec_start = int(metadata['second'])

    w = AudioSegment.from_wav(f)
    w = helpers.remove_dc_offset(w)

    data_hr = AudioSegment.silent(duration=3600 * 1000, frame_rate=w.frame_rate)
    data_hr = data_hr.overlay(w, position=sec_start * 1000)
    # hr_result.export(f'annotation/_output/test_{hour_start}.wav', format='wav')
    data = data + data_hr

# Save 1 minute subsamples to file
for t in sample_times:
    t_outpath = f'{output}/{id}_{year}{month:02d}{day:02d}_{t.hour:02d}{t.minute:02d}{t.second:02d}_SS.wav'
    print(t_outpath)
    start_time_ms = (t.hour * 60 * 60 + t.minute * 60 + t.second) * 1000
    end_time_ms = start_time_ms + 60 * 1000 # one minute subsample
    data_subsample = data[start_time_ms:end_time_ms]
    data_subsample.export(t_outpath, format='wav')
