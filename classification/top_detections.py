## Get the top N most confident detections per species at each unique site

# Manually run this script for each site, changing in_dir as needed
top_dir = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data'
in_dir  = top_dir + '/raw_detections/2020/Deployment1/S4A04271_20200412_Data' # CHANGE ME!
out_dir = top_dir + '/top_detections'

import os
import pandas as pd

N = 5

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

# Display the grouped DataFrame
top_N_detections = all_detections.groupby('common_name').apply(lambda x: x.nlargest(N, 'confidence')).reset_index(drop=True)
print(top_N_detections)


# save top_N_detections to file under a 'top_detections' folder
# print(top_dir + '/top_detections')
path_out = out_dir + os.path.splitext(in_dir[len(top_dir + '/raw_detections'):])[0] + '.csv'
print(f'Saving to {path_out}')
if not os.path.exists(os.path.dirname(path_out)):
    os.makedirs(os.path.dirname(path_out))
pd.DataFrame.to_csv(top_N_detections, path_out, index=False) 

# NEXT: see extract_detection_audio.py