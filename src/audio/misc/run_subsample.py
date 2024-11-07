# Split a specified site-day into 1-minute subsamples

# NOTE: Change these values manually for each run
in_dir  = '/Volumes/gioj/OESF/2020/Deployment5/SMA00380_20200604'
# in_dir = '/Users/giojacuzzi/Desktop/audio_test/SMA00309_20230530_000000'
lat   = 47.67497
lon   = -124.13689
year  = 2020
month_start = 6
month_end   = 6
day_start   = 4
day_end     = 13
import os

# outpath = '/Users/giojacuzzi/Desktop/audio_test/fudge'
outpath = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator'
id    = os.path.basename(in_dir).split('_')[0]

# ---------------------------------------------------------------------
from subsample import *
import numpy as np
from pydub import AudioSegment
import tools
from birdnetlib.analyzer import Analyzer
import classification.run_analyze_dir_parallel as run_analyze_dir_parallel
import sys

species_list_path = os.path.abspath('classification/species_list/species_list_OESF.txt')

# Configure analyzer for predictions
sort_by = 'confidence'
ascending = False
n_processes = 8 # cores per batch
min_confidence = 0.0
num_separation = 4
cleanup = True

# Run subsample routine
if __name__ == '__main__':

    # Create output directory
    output = outpath + f'/{id}_{year}{month_start:02d}{day_start:02d}'
    if not os.path.exists(output):
        os.makedirs(output)

    # Calculate subsample times
    sample_times = get_subsample_datetimes(id, lat, lon, year, month_start, month_end, day_start, day_end)
    for s in sample_times:
        print(s.strftime("%Y-%m-%d %H:%M:%S"))

    # Build an excel spreadsheet dataframe for logging notes
    subsample_files = []

    # Create subsample wavs for each date
    for date in pd.date_range(start=datetime(year=year, month=month_start, day=day_start), end=datetime(year=year, month=month_end, day=day_end)):
        print(f'Subsampling date {date.strftime("%Y-%m-%d")}...')
        datetimes_with_same_date = [time for time in sample_times if time.date() == date.date()]

        # Combine recordings from day into single data structure
        files = find_files_by_date(in_dir, date.year, date.month, date.day)

        if len(files) == 0:
            print(f'No files found for date {date}')
            continue

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

        # Build an excel spreadsheet dataframe for logging notes
        date_subsample_files = []

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

            # add t_outpath to the spreadsheet under 'file' column
            date_subsample_files.append(os.path.basename(t_outpath))
        
        subsample_files = subsample_files + date_subsample_files

    # Save an excel spreadsheet
    subsample_files.sort()
    xlsx_log = pd.DataFrame({
        'file': subsample_files,
        'annotator': [''] * len(subsample_files),
        'annotator_notes': [''] * len(subsample_files),
        'reviewer': [''] * len(subsample_files),
        'reviewer_notes': [''] * len(subsample_files)
    })
    path_out = output + '/log_' + os.path.basename(output) + '.xlsx'
    print(f'Saving xlsx log to {path_out}')
    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))
    pd.DataFrame.to_excel(xlsx_log, path_out, index=False)

    # # use classifier with sound separation to make predictions for reference
    # print(f'Analyzing subsamples in in {outpath}')
    # run_analyze_parallel.init_global_analyzer()
    # run_analyze_parallel.process_dir_parallel(
    #     in_dir = outpath,
    #     out_dir = outpath + '/predictions',
    #     in_filetype='.wav',
    #     n_processes=7,
    #     min_confidence=0.0,
    #     num_separation=4,
    #     cleanup=True,
    #     sort_by='confidence',
    #     ascending=False
    # )
    # print('Done!')
    
