# Given a deployment number, site ID, and serial number, review audio for raw classifier detections, sorted by descending confidence score

# CHANGE ME
deployment = 8
serial_no  = "SMA00424"
label = "northern goshawk"

threshold = 0.1

path_detections = f'/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/transfer learning/data/raw_detections/2020/Deployment{deployment}'
path_audio = f'/Volumes/gioj_b1 1/OESF/2020/Deployment{deployment}'
path_out = '/Users/giojacuzzi/Downloads/review_site_label_detections'

import os
import fnmatch
import pandas as pd
import sys
from datetime import datetime, timedelta
from pydub import AudioSegment
import shutil
from utils.files import *
from utils.audio import *
import subprocess

if os.path.exists(path_out):
    shutil.rmtree(path_out)
os.makedirs(path_out)

def find_subdirectory_with_substring(directory, substring):
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if substring in dir_name:
                return os.path.join(root, dir_name)
    return None

dir = find_subdirectory_with_substring(path_detections, serial_no)

if dir == None:
    print(f'ERROR: Could not find {dir}')
    sys.exit()

print(f'Loading all .csv files at {dir}...')

csv_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.csv')]
dfs = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)

print(f'Filtering for label {label}...')
combined_df['common_name'] = combined_df['common_name'].str.lower()
detections = combined_df[combined_df['common_name'] == label]
detections = detections[detections['confidence'] > threshold]
detections = detections.sort_values(by="confidence", ascending=False)
detections['confidence'] = round(detections['confidence'], 2)

print(detections)
print(detections.shape)

def find_matching_file(files, target_datetime):
    for file in files:
        # Extract datetime from filename
        file_datetime_str = os.path.basename(file).split('_')[1] + os.path.basename(file).split('_')[2].replace('.wav', '')
        file_datetime = datetime.strptime(file_datetime_str, '%Y%m%d%H%M%S')
        
        # Check if the target datetime is within the 1-hour range of the file
        if file_datetime <= target_datetime < file_datetime + timedelta(hours=1):
            delta_sec =  (target_datetime - file_datetime).total_seconds()
            print(f'delta_sec {delta_sec}')
            print(f'Extracting audio data {file} (time delta {(target_datetime - file_datetime)})')

            # Load the entire audio file
            audio_data = AudioSegment.from_wav(os.path.abspath(file))
            audio_data = remove_dc_offset(audio_data)

            # TODO: Extract the audio data and save to file
            extracted_data = audio_data[(delta_sec * 1000):((delta_sec + 3) * 1000)]
            full_out = f'{path_out}/{serial_no}_{round(confidence, 4)}_{i}.wav'
            if not os.path.exists(path_out):
                os.makedirs(path_out)
            print(f'Saving to {full_out}')
            extracted_data.export(full_out, format='wav')
            subprocess.run(["open", full_out])
            return
    print(f'ERROR: Could not find file for {target_datetime}')

print(f'Reviewing detections:')
i = 1
for index, row in detections.iterrows():
    print(f"{i}/{len(detections)} ({len(detections) - i} remaining)")
    confidence = row['confidence']
    print(f"confidence: {confidence}")
    specific_datetime = datetime.strptime(row['start_date'], '%Y-%m-%d %H:%M:%S')
    print(f"{specific_datetime}")

    # TODO: Extract associated audio, save to file, and open with default application
    path_detection_audio = f'{path_audio}/{os.path.basename(dir)}'

    audio_files = [os.path.join(path_detection_audio, f) for f in os.listdir(path_detection_audio) if f.endswith('.wav')]
    find_matching_file(audio_files, specific_datetime)

    input('Press [return] for next detection...')
    i += 1
