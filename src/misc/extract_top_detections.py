# Get the top detections for a specific species

species_to_evaluate = 'all'
species_to_evaluate = 'Sooty Grouse'

# Manually run this script for each site, changing in_dir as needed
top_dir = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data'
in_dir  = top_dir + '/raw_detections/2020/Deployment1' # CHANGE ME
out_dir = '/Users/giojacuzzi/Desktop/TESTY'
data_dir = '/Volumes/gioj_b1/OESF' # CHANGE ME

threshold = 0.1 # only sample from detections above (to filter out "zero-inflated" silence)
extract_audio_files = True
import os
import pandas as pd
from utils.files import *
from utils.labels import *
from annotation.extract_detection_audio import *
from datetime import timedelta
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)  # Display full width of the DataFrame


dir_out = out_dir
if not os.path.exists(dir_out):
    os.makedirs(dir_out)


species = get_species_classes()
if species_to_evaluate == 'all':
    species_to_evaluate = sorted(species)

# combine all the detections at that site
# Initialize an empty DataFrame to store the combined data
all_detections = pd.DataFrame()

# Loop through each .csv file and concatenate its data to the combined DataFrame
# files = [file for file in os.listdir(in_dir) if file.endswith('.csv')]
files = find_files(in_dir, suffix='.csv')
print(f'Loading species detections among {len(files)} files...')
for file in files:
    print(file)
    file_path = os.path.join(in_dir, file)
    detections = pd.read_csv(file_path)
    detections['file'] = os.path.basename(os.path.splitext(file_path)[0])

    if not detections.empty:
        species_detections = detections[detections['common_name'] == species_to_evaluate]
        # print(species_detections.groupby('common_name')['confidence'].agg(['min', 'max']))

        if not species_detections.empty:
            all_detections = pd.concat([all_detections, species_detections], ignore_index=True)

print(f'Loaded all {len(all_detections)} detections')
print(all_detections.head())

all_detections = all_detections[all_detections['confidence'] > threshold]

all_detections = all_detections.sort_values(by=['confidence'], ascending=[False])
print(all_detections.to_string())

if extract_audio_files:
    # Now extract audio files for each detection
    for index, row in all_detections.iterrows():
        common_name = row['common_name']
        confidence  = row['confidence']
        start_date  = row['start_date']
        # end_date    = row['end_date']
        file = os.path.basename(row['file']) + '.wav'

        print(common_name)
        print(confidence)
        print(start_date)
        # print(end_date)
        print(file)

        # Find the original audio file matching 'file' under 'root_dir'
        print('Finding audio file path...')
        file_path = find_file_full_path(data_dir, file)
        print(file_path)

        date_format = '%Y-%m-%d %H:%M:%S'
        date_detection_start = datetime.strptime(start_date, date_format)
        # date_detection_end = datetime.strptime(end_date, date_format)
        extract_detection_audio(file_path, dir_out, date_detection_start, date_detection_start + timedelta(seconds=3), tag=f'{common_name}-{confidence}')
    
    print('Finished extracting all detections as audio files!')