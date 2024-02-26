top_dir = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data'
in_dir  = top_dir + '/raw_detections/2020/Deployment1/S4A04271_20200412_Data' # CHANGE ME!
out_dir = top_dir + '/top_detections'

import os
import pandas as pd

# combine all the detections at that site
# Initialize an empty DataFrame to store the combined data
all_detections = pd.DataFrame()

# Loop through each .csv file and concatenate its data to the combined DataFrame
files = [file for file in os.listdir(in_dir) if file.endswith('.csv')]
print(f'Found {len(files)} files...')
for file in files:
    file_path = os.path.join(in_dir, file)
    detections = pd.read_csv(file_path)
    detections['file'] = os.path.splitext(file)[0]
    if not detections.empty:
        all_detections = pd.concat([all_detections, detections], ignore_index=True)

# Display the combined DataFrame
print(all_detections)

# 