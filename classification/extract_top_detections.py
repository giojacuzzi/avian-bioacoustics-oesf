## Get the top N most confident detections per species at each unique site
#
## Create a series of top N validation subsamples and corresponding spreadsheets
# For a given site deployment...
#
# create 3 sec audio subsamples for each of the top N detections using
# top_N_detections.py
# extract_subsample.py
#
# create a spreadsheet that lists each subsample file along with the: predicted species, validated species, Reviewer 1, and Reviewer 2

# Manually run this script for each site, changing in_dir as needed
top_dir = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data'
in_dir  = top_dir + '/raw_detections/2020/Deployment1/SMA00351_20200412_Data' # CHANGE ME!
out_dir = top_dir + '/top_detections'
data_dir = '/Volumes/gioj_b1/OESF'

extract_audio_files = True

import os
import pandas as pd
import sys
from helper import *
from extract_detection_audio import *

N = 5

# combine all the detections at that site
# Initialize an empty DataFrame to store the combined data
all_detections = pd.DataFrame()

# Loop through each .csv file and concatenate its data to the combined DataFrame
files = [file for file in os.listdir(in_dir) if file.endswith('.csv')]
print(f'Finding the top {N} detections per species among {len(files)} files...')
for file in files:
    file_path = os.path.join(in_dir, file)
    detections = pd.read_csv(file_path)
    detections['file'] = os.path.basename(os.path.splitext(file_path)[0])
    if not detections.empty:
        all_detections = pd.concat([all_detections, detections], ignore_index=True)

# Display the combined DataFrame
print(all_detections)

# Display the grouped DataFrame
top_N_detections = all_detections.groupby('common_name').apply(lambda x: x.nlargest(N, 'confidence')).reset_index(drop=True)
top_N_detections.rename(columns={'common_name': 'detected'}, inplace=True)
top_N_detections.insert(0, 'validated', '')
print(top_N_detections)

# Save top_N_detections as excel spreadsheet under a 'top_detections' folder
dir_out = out_dir + os.path.splitext(in_dir[len(top_dir + '/raw_detections'):])[0]
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
path_out = dir_out + '/' + os.path.basename(dir_out) + '.xlsx'
print(f'Saving to {path_out}')
if not os.path.exists(os.path.dirname(path_out)):
    os.makedirs(os.path.dirname(path_out))
pd.DataFrame.to_excel(top_N_detections, path_out, index=False)

if extract_audio_files:
    # Now extract audio files for each detection
    for index, row in top_N_detections.iterrows():
        common_name = row['detected']
        confidence  = row['confidence']
        start_date  = row['start_date']
        end_date    = row['end_date']
        file = os.path.basename(row['file']) + '.wav'

        print(common_name)
        print(confidence)
        print(start_date)
        print(end_date)
        print(file)

        # Find the original audio file matching 'file' under 'root_dir'
        print('Finding audio file path...')
        file_path = find_file_full_path(data_dir, file)
        print(file_path)

        date_format = '%Y-%m-%d %H:%M:%S'
        date_detection_start = datetime.datetime.strptime(start_date, date_format)
        date_detection_end   = datetime.datetime.strptime(end_date, date_format)
        extract_detection_audio(file_path, dir_out, date_detection_start, date_detection_end, tag=f'{common_name}-{confidence}')
    
    print('Finished extracting all detections as audio files!')
