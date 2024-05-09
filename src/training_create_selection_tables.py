from annotation.annotations import *
from utils.log import *
import pandas as pd
import os
import sys

label_truth = "northern pygmy-owl" # i.e. label_truth
species_predicted = "northern pygmy-owl" # or another, e.g. "wilson's warbler" vs "pacific wren"

# output_path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/training_data'
output_path = '/Users/giojacuzzi/Downloads'

# Get raw annotation data
raw_annotations = get_raw_annotations()

# Collate raw annotation data into species detection labels per species
if label_truth == species_predicted:
    print("Finding correct detection annotation examples for the species alone...")
    collated_detection_labels = collate_annotations_as_detections(raw_annotations, [label_truth], only_annotated=True)
    print(collated_detection_labels.to_string())
    collated_detection_labels = collated_detection_labels[(collated_detection_labels['label_truth'] == label_truth)] 
else:
    print("Finding both annotation examples for any detections among any species...")
    collated_detection_labels = raw_annotations[(raw_annotations['label_truth'] == label_truth) & (raw_annotations['species_predicted'] == species_predicted)]

print(collated_detection_labels.to_string())

# Extract data for annotation...
raw_metadata = files.get_raw_metadata()
raw_metadata['date']     = pd.to_datetime(raw_metadata['date'], format='%Y%m%d')
raw_metadata['time']     = pd.to_timedelta(raw_metadata['time'].astype(str).str.zfill(6).str[:2] + ':' +
                                           raw_metadata['time'].astype(str).str.zfill(6).str[2:4] + ':' +
                                           raw_metadata['time'].astype(str).str.zfill(6).str[4:])
# print(raw_metadata)

# random_detection = collated_detection_labels.sample(n=1)
# print(random_detection.to_string())

for index, detection in collated_detection_labels.iterrows():
    print(detection)
    species_predicted = detection['species_predicted']
    conf = detection['confidence']

    # Find the raw audio file that contains the detection

    # Convert date to datetime, time to timedelta
    detection['date'] = pd.to_datetime(detection['date'], format='%Y%m%d')
    detection['time'] = pd.to_timedelta(str(detection['time']).zfill(6)[:2] + ':' +
                                            str(detection['time']).zfill(6)[2:4] + ':' +
                                            str(detection['time']).zfill(6)[4:])

    # Merge, then filter rows where the detection time falls within  range of audio file start and end
    merged_df = pd.merge(pd.DataFrame([detection]), raw_metadata, on=['serialno', 'date'], how='inner')
    merged_df = merged_df[(merged_df['time_x'] >= merged_df['time_y']) & # time_x is the start time of the detection, time_y of the hour-long file
                        (merged_df['time_x'] <= merged_df['time_y'] + pd.Timedelta(hours=1))]  # NOTE: 1 hour file duration
    merged_df = merged_df.iloc[0]

    print(merged_df.transpose().to_string())
    time_delta = (merged_df['time_x'] - merged_df['time_y'])
    print(f"time delta: {time_delta}")

    # Create a Raven Pro selection table
    selection_table = pd.DataFrame({
        'Selection':      [1],
        'View':           ['Spectrogram 1'],
        'Channel':        [1],
        'Begin Time (s)': [time_delta.total_seconds()],
        'End Time (s)':   [time_delta.total_seconds() + 3.0],
        'Low Freq (Hz)':  [0.0],
        'High Freq (Hz)': [16000.0],
        'Species':        [label_truth],
        'Type':           ['']
    })
    print(selection_table)
    print(os.path.basename(merged_df["filepath"]))

    file = f'{os.path.splitext(os.path.basename(merged_df["filepath"]))[0]}.Table.1.selections.txt'
    path = f'{output_path}/{label_truth}/{species_predicted}/{conf}'
    if not os.path.exists(path):
        os.makedirs(path)

    filepath = f'{path}/{file}'
    selection_table.to_csv(filepath, sep='\t', index=False)
    print_success(f'Saved {filepath}')
