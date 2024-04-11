from annotation.annotations import *
from utils.log import *
import pandas as pd
import os

species_to_annotate = 'pacific-slope flycatcher'

output_path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/training_data'

# Get raw annotation data
raw_annotations = get_raw_annotations()

# Collate raw annotation data into species detection labels per species
collated_detection_labels = collate_annotations_as_detections(raw_annotations, [species_to_annotate], only_annotated=True)
collated_detection_labels = collated_detection_labels[collated_detection_labels['label_truth'] == species_to_annotate]

print(collated_detection_labels.to_string())

random_detection = collated_detection_labels.sample(n=1)
# print(random_detection.to_string())

# TODO: extract data for annotation...
raw_metadata = files.get_raw_metadata()
# print(raw_metadata)

# Find the raw audio file that contains the detection

# Convert date to datetime, time to timedelta
random_detection['date'] = pd.to_datetime(random_detection['date'], format='%Y%m%d')
raw_metadata['date']     = pd.to_datetime(raw_metadata['date'], format='%Y%m%d')
random_detection['time'] = pd.to_timedelta(random_detection['time'].astype(str).str.zfill(6).str[:2] + ':' +
                                           random_detection['time'].astype(str).str.zfill(6).str[2:4] + ':' +
                                           random_detection['time'].astype(str).str.zfill(6).str[4:])
raw_metadata['time']     = pd.to_timedelta(raw_metadata['time'].astype(str).str.zfill(6).str[:2] + ':' +
                                           raw_metadata['time'].astype(str).str.zfill(6).str[2:4] + ':' +
                                           raw_metadata['time'].astype(str).str.zfill(6).str[4:])

# Merge, then filter rows where the detection time falls within  range of audio file start and end
merged_df = pd.merge(random_detection, raw_metadata, on=['serialno', 'date'], how='inner')
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
    'Species':        [species_to_annotate]
})
print(selection_table)
print(os.path.basename(merged_df["filepath"]))

filepath = f'{output_path}/{species_to_annotate}/{os.path.splitext(os.path.basename(merged_df["filepath"]))[0]}.Table.1.selections.txt'

selection_table.to_csv(filepath, sep='\t', index=False)
print_success(f'Saved {filepath}')
