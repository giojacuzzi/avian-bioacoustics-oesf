from subsample_time import *
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

# result = AudioSegment.silent(24 * 3600 * 1000, frame_rate=AudioSegment.from_wav(files[0]).frame_rate)
res = AudioSegment.empty()

def remove_dc_offset(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    samples = samples - round(np.mean(samples))
    return(AudioSegment(samples.tobytes(), channels=audio_segment.channels, sample_width=audio_segment.sample_width, frame_rate=audio_segment.frame_rate)) 

for f in files[:2]:
    print(f)
    metadata = get_metadata_from_filename(f)
    hour_start = int(metadata['hour'])
    sec_start = int(metadata['second'])
    # print(metadata['hour'], metadata['minute'], metadata['second'])

    w_data = AudioSegment.from_wav(f)
    w_data = remove_dc_offset(w_data)

    hr_result = AudioSegment.silent(duration=3600 * 1000, frame_rate=w_data.frame_rate)
    hr_result = hr_result.overlay(w_data, position=sec_start * 1000)
    # hr_result.export(f'annotation/_output/test_{hour_start}.wav', format='wav')
    res = res + hr_result

res.export('annotation/_output/res.wav', format='wav')

# from pydub import audio_segment
# # w0.get_array_of_samples()