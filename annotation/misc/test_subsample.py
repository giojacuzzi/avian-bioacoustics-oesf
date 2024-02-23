from annotation.subsample import *
import numpy as np
from pydub import AudioSegment

# get_metadata_from_filename
path = '/Users/giojacuzzi/Desktop/audio_test/SMA00351_20210502_000000/SMA00351_20210502_050002.wav'
# print(get_metadata_from_filename(path))

# Get subsamples for entire day
id = '?'
lat = 47.64482
lon = -124.19401
year = 2021
month = 5
day = 2
sample_times = subsample(id, lat, lon, year, month, day)
# print(sample_times[0])

# Merge audio for entire day
directory = '/Users/giojacuzzi/Desktop/audio_test/SMA00351_20210502_000000'
os.listdir(directory)
files = find_files_by_date(directory, year, month, day)

data = AudioSegment.empty()

def remove_dc_offset(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    samples = samples - round(np.mean(samples))
    return(AudioSegment(samples.tobytes(), channels=audio_segment.channels, sample_width=audio_segment.sample_width, frame_rate=audio_segment.frame_rate)) 

id = get_metadata_from_filename(files[0])['id']

for f in files:
    print(f)
    metadata = get_metadata_from_filename(f)
    hour_start = int(metadata['hour'])
    sec_start = int(metadata['second'])
    # print(metadata['hour'], metadata['minute'], metadata['second'])

    w = AudioSegment.from_wav(f)
    w = remove_dc_offset(w)

    data_hr = AudioSegment.silent(duration=3600 * 1000, frame_rate=w.frame_rate)
    data_hr = data_hr.overlay(w, position=sec_start * 1000)
    # hr_result.export(f'annotation/_output/test_{hour_start}.wav', format='wav')
    data = data + data_hr

# data.export('annotation/_output/data.wav', format='wav')

for t in sample_times:
    print(t)
    start_time_ms = (t.hour * 60 * 60 + t.minute * 60 + t.second) * 1000
    end_time_ms = start_time_ms + 60 * 1000 # one minute subsample
    data_subsample = data[start_time_ms:end_time_ms]
    data_subsample.export(f'annotation/_output/ss_{id}_{year}{month:02d}{day:02d}_{t.hour:02d}{t.minute:02d}{t.second:02d}.wav', format='wav')
