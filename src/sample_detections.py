## Sample N random detections per species from site deployment data

# Manually run this script for each site, changing in_dir as needed
top_dir = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data'
in_dir  = top_dir + '/raw_detections/2020/Deployment2/SMA00351_20200424_Data' # CHANGE ME
out_dir = top_dir + '/_annotator'
data_dir = '/Volumes/gioj_b1/OESF' # CHANGE ME

N = 6 # number of samples to take
threshold = 0.1 # only sample from detections above (to filter out "zero-inflated" silence)
extract_audio_files = True

import os
import pandas as pd
from utils.files import *
from annotation.extract_detection_audio import *
from datetime import timedelta
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)  # Display full width of the DataFrame

# combine all the detections at that site
# Initialize an empty DataFrame to store the combined data
all_detections = pd.DataFrame()

# Loop through each .csv file and concatenate its data to the combined DataFrame
files = [file for file in os.listdir(in_dir) if file.endswith('.csv')]
print(f'Loading species detections among {len(files)} files...')
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
# print(all_detections.head())

print(all_detections.groupby('common_name')['confidence'].agg(['min', 'max']))

# # Plot distribution of scores for each species
# for common_name in all_detections['common_name'].unique():
#     subset = all_detections[all_detections['common_name'] == common_name]
#     bin_width = 0.01
#     bins = np.arange(subset['confidence'].min(), subset['confidence'].max() + bin_width, bin_width)
#     plt.hist(subset['confidence'], bins=bins, color='skyblue', edgecolor='black')
#     plt.xlabel('Confidence')
#     plt.ylabel('Frequency')
#     plt.title(f'Binned Histogram of Confidence for {common_name}')
#     plt.show()

# For each species, select N detections at random.
print(f'Selecting {N} random samples per species...')

# We will pull available detections from the pool of all detections
available_detections = all_detections[all_detections['confidence'] > threshold]

# Function to sample N rows from each group without replacement.
# Also intentionally includes the sample with the maximum confidence.
def sample_group(group, N):
    sampled = group.sample(min(len(group), N))
    max_confidence_row = group.loc[group['confidence'].idxmax()]
    if max_confidence_row.name not in sampled.index:
        sampled = pd.concat([sampled, max_confidence_row.to_frame().T])
    stats = {
        'min_confidence': group['confidence'].min(),
        'max_confidence': group['confidence'].max(),
        'avg_confidence': group['confidence'].mean()
    }
    print(f"Common Name: {group['common_name'].iloc[0]}")
    print("Sampled Rows:")
    print(sampled)
    print("Statistics:")
    print(stats)
    print()
    return sampled

# Apply the sampling function to each group
detection_samples = available_detections.groupby('common_name', group_keys=False).apply(sample_group, N=N)
# detection_samples = detection_samples.sort_values(by='common_name')
print(detection_samples)

# Save detection_samples as excel spreadsheet
detection_samples # add columns for annotator and reviewer
detection_samples.insert(1, 'reviewer', '')
detection_samples.insert(1, 'reviewer_notes', '')
detection_samples.insert(1, 'annotator', '')
detection_samples.insert(1, 'annotator_notes', '')
dir_out = out_dir + os.path.splitext(in_dir[len(top_dir + '/raw_detections'):])[0]
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
path_out = dir_out + '/_annotations_' + os.path.basename(dir_out) + '.xlsx'
print(f'Saving to {path_out}')
if not os.path.exists(os.path.dirname(path_out)):
    os.makedirs(os.path.dirname(path_out))
pd.DataFrame.to_excel(detection_samples, path_out, index=False)

if extract_audio_files:
    # Now extract audio files for each detection
    for index, row in detection_samples.iterrows():
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
