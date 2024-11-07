## Get N random detections per species at each unique site
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
in_dir  = top_dir + '/raw_detections/2020/Deployment4/SMA00410_20200523' # CHANGE ME!
out_dir = top_dir + '/annotated_detections'
data_dir = '/Volumes/gioj/OESF'

# in_dir = '/Users/giojacuzzi/Desktop/audio_test/detections'
# out_dir = in_dir + '/out'

extract_audio_files = True

import os
import pandas as pd
import numpy as np
import sys
from tools import *
from extract_detection_audio import *
from datetime import timedelta
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)  # Display full width of the DataFrame

N = 10

# combine all the detections at that site
# Initialize an empty DataFrame to store the combined data
all_detections = pd.DataFrame()

# Loop through each .csv file and concatenate its data to the combined DataFrame
files = [file for file in os.listdir(in_dir) if file.endswith('.csv')]
print(f'Finding the top {N} detections per species among {len(files)} files...')
for file in files:
    print(file)
    file_path = os.path.join(in_dir, file)
    detections = pd.read_csv(file_path)
    detections['file'] = os.path.basename(os.path.splitext(file_path)[0])
    if not detections.empty:
        all_detections = pd.concat([all_detections, detections], ignore_index=True)

# Display the combined DataFrame
# print(all_detections)
print('Loaded all detections')

# For each species, select N detections at random. Rather than choosing randomly from
# the population of detections (which would result in a disproportionate number of very
# low-scoring detections due to the majority of silence), pull values from a uniform
# distribution that ranges from the minimum observed score to the maximum, and choose those
# detections that have the nearest scores.
print(f'Selecting {N} random samples per species...')
# Get the minimum and maximum scores for each species
species_scores = all_detections.groupby('common_name')['confidence'].agg(['min', 'max'])
print(species_scores)
# sys.exit()

threshold = 0.01 # USE A THRESHOLD

def generate_random_numbers(row):
    min_val = max(threshold, row['min'])
    max_val = row['max']
    return np.random.uniform(min_val, max_val, N)

# Generate random scores
species_scores['random_scores'] = species_scores.apply(generate_random_numbers, axis=1)
# print(species_scores)

# We will pull available detections from the pool of all detections
available_detections = all_detections[all_detections['confidence'] >= threshold]


# Create an empty DataFrame to store the removed rows
detection_samples = pd.DataFrame()

# Iterate over each unique common_name in score_bounds
for common_name, group in species_scores.groupby('common_name'):
    # print(f"{common_name}")
    # Iterate over each random score associated with the current common_name
    for random_scores in group['random_scores']:
        # Iterate over each random score
        for score in random_scores:
            # print(f"Score: {score}")
            # Find the closest confidence value in available_detections
            filtered_detections = available_detections[available_detections['common_name'] == common_name]
            if filtered_detections.empty:
                # print(f'Ran out of detections for {common_name}. Moving on...')
                continue

            closest_row_index = np.abs(filtered_detections['confidence'] - score).idxmin()
            # Save the row into detection_samples DataFrame
            # print(available_detections.loc[[closest_row_index]])
            detection_samples = pd.concat([detection_samples, available_detections.loc[[closest_row_index]]], ignore_index=True)
            # Remove the row from available_detections
            available_detections = available_detections.drop(index=closest_row_index, inplace=False)

# Reset index of available_detections
available_detections.reset_index(drop=True, inplace=True)

# Reset index of detection_samples
detection_samples.reset_index(drop=True, inplace=True)

detection_samples = detection_samples.sort_values(by='common_name')
print(detection_samples)

# # Save detection_samples as excel spreadsheet under a 'annotated_detections' folder
# dir_out = out_dir + os.path.splitext(in_dir[len(top_dir + '/raw_detections'):])[0]
# if not os.path.exists(dir_out):
#     os.makedirs(dir_out)
# path_out = dir_out + '/' + os.path.basename(dir_out) + '.xlsx'
# print(f'Saving to {path_out}')
# if not os.path.exists(os.path.dirname(path_out)):
#     os.makedirs(os.path.dirname(path_out))
# pd.DataFrame.to_excel(detection_samples, path_out, index=False)

# if extract_audio_files:
#     # Now extract audio files for each detection
#     for index, row in detection_samples.iterrows():
#         common_name = row['detected']
#         confidence  = row['confidence']
#         start_date  = row['start_date']
#         # end_date    = row['end_date']
#         file = os.path.basename(row['file']) + '.wav'

#         print(common_name)
#         print(confidence)
#         print(start_date)
#         # print(end_date)
#         print(file)

#         # Find the original audio file matching 'file' under 'root_dir'
#         print('Finding audio file path...')
#         file_path = find_file_full_path(data_dir, file)
#         print(file_path)

#         date_format = '%Y-%m-%d %H:%M:%S'
#         date_detection_start = datetime.datetime.strptime(start_date, date_format)
#         # date_detection_end   = datetime.datetime.strptime(end_date, date_format)
#         extract_detection_audio(file_path, dir_out, date_detection_start, date_detection_start + timedelta(seconds=3), tag=f'{common_name}-{confidence}')
    
#     print('Finished extracting all detections as audio files!')
