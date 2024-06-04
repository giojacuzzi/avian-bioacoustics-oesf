## Extract annotated audio examples as training data for a custom model

from utils.files import *
from utils.audio import *
from utils import labels
import string
import os
import shutil
from pydub import AudioSegment
import sys

# Path to directory that contains the annotated Raven Pro selection tables
dir_training_input_annotations = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/training_data'
# Path to directory that will be populated with training data
dir_training_output_data = '/Users/giojacuzzi/repos/avian-bioacoustics-oesf/data/training/Custom'
# Path to directory with raw audio data
data_dir = '/Volumes/gioj_b1/OESF'

overwrite = True

# Given a list of labels to train...
species_labels = [
    "pacific-slope flycatcher",
    "varied thrush",
    "wilson's warbler",
    "marbled murrelet",
    "hammond's flycatcher",
    "red-breasted nuthatch",
    "northern saw-whet owl",
    "northern pygmy-owl" 
]
labels_to_train = species_labels + [
    "abiotic vehicle"
] # e.g. 'abiotic vehicle' or others. Note that "Background" class is unique and automatically searched for.

species_list = pd.read_csv(species_list_filepath, index_col=None, usecols=['common_name', 'scientific_name'])
species_list['common_name']     = species_list['common_name'].str.lower()
species_list['scientific_name'] = species_list['scientific_name'].str.lower()
print(f'common names: {species_list["common_name"].to_string()}')

if overwrite and os.path.exists(dir_training_output_data):
    shutil.rmtree(dir_training_output_data)

training_examples = pd.DataFrame()

# For each label...
for label in labels_to_train:

    print(f'Processing label "{label}"')

    # Format the label name to match BirdNET
    if label in species_list['common_name'].values: # Biotic label
        row = species_list.loc[species_list['common_name'] == label]
        if not row.empty:
            scientific_name = row.iloc[0]['scientific_name']
            label_label = f"{scientific_name.capitalize()}_{string.capwords(label)}" # required BirdNET format
        else:
            print_error(f"Scientific name for {label} not found.")
            continue
    else: # Abiotic label
        label_label = f"Abiotic_{string.capwords(label)}"

    # Get the associated annotations for the label
    dir_label_input_annotations = dir_training_input_annotations + '/' + label
    annotation_files = find_files(dir_label_input_annotations, 'selections.txt')
    if len(annotation_files) == 0:
        print_error(f'No files found for label "{label}"')
        continue

    # Use the annotations to extract examples from the raw audio data
    for file in annotation_files:

        print(f'Loading annotation file {file}...')

        # Parse metadata from file
        metadata = parse_metadata_from_filename(file)
        print(metadata)
        
        # Load the annotations
        table = load_raven_selection_table(file, cols_needed=['selection', 'label', 'type', 'begin file', 'file offset (s)', 'delta time (s)', 'low freq (hz)', 'high freq (hz)'])

        # Remove annotation references (these are used in the test dataset)
        table = table[(table['low freq (hz)'] != 0.0) & (table['high freq (hz)'] != 16000.0)]

        # Further validate the annotations
        if (table['label'] == 'nan').any() or table['label'].isnull().any():
            print_warning(f'{os.path.basename(file)} contains annotations with missing labels')
        table['label'] = table['label'].str.lower()
        table['label'] = table['label'].apply(labels.clean_label)
        table_errors = table[table['delta time (s)'] > 3.0]
        for index, row in table_errors.iterrows():
            print_error(f"Selection {row['selection']} extends beyond 3 seconds in {file}")

        table['selection_start_time'] = table['file offset (s)']
        table['selection_end_time']   = table['file offset (s)'] + table['delta time (s)']

        # Extract the raw audio data to the training data directory
        # print(dir_training_output_data)
        for index, row in table.iterrows():
            print(f"Extracting audio data for row {index} ({row['label']})...")

            if row['label'] != label:
                print('Skipping non-label annotation...')
                continue

            annotation_start_time = row['file offset (s)']
            annotation_end_time = annotation_start_time + row['delta time (s)']
            midpoint = annotation_start_time + (annotation_end_time - annotation_start_time)/2.0

            # Create a 3-second window centered at the midpoint of the annotation
            data_start_time = max(midpoint - 3.0/2.0, 0.0)
            data_end_time = max(data_start_time + 3.0, annotation_end_time) # annotation_end_time is guaranteed to be <= the end of the raw audio data

            file_stub = f"{metadata['serial_no']}_{metadata['date']}_{metadata['time']}_{data_start_time}"
            file_out_audio = f"{file_stub}.wav"
            file_out_selections = f"{file_stub}.Table.1.selections.txt"

            # Find all other annotations within this window and save to a raven pro selection table txt

            # Filter the selections that overlap with the annotation period
            overlap_condition = (
                (table['selection_start_time'] < annotation_end_time) &
                (table['selection_end_time']   > annotation_start_time)
            )
            overlapping_selections = table[overlap_condition]
            overlapping_selections = overlapping_selections.copy()
            overlapping_selections.loc[:, 'file offset (s)'] = 0.0
            overlapping_selections.loc[:, 'delta time (s)'] = 3.0
            overlapping_selections.loc[:, 'file_audio'] = file_out_audio
            path_out = f'{dir_training_output_data}/selections/{label_label}'
            full_out = f'{path_out}/{file_out_selections}'
            if not os.path.exists(path_out):
                os.makedirs(path_out)
            print(f'Saving to {full_out}')
            overlapping_selections.to_csv(full_out, sep='\t', index=False)

            # Find the original audio file matching 'file' under 'root_dir'
            audio_file = find_file_full_path(data_dir, row['begin file'])
            if audio_file is None:
                print_error("Could not find corresponding .wav file")
                continue

            # Load the entire audio file
            audio_data = AudioSegment.from_wav(audio_file)
            audio_data = remove_dc_offset(audio_data)

            # Extract the audio data and save to file
            extracted_data = audio_data[(data_start_time * 1000):(data_end_time * 1000)]
            path_out = f'{dir_training_output_data}/audio/{label_label}'
            full_out = f'{path_out}/{file_out_audio}'
            if not os.path.exists(path_out):
                os.makedirs(path_out)
            print(f'Saving to {full_out}')
            extracted_data.export(full_out, format='wav')

        # Store info for later summary
        training_examples = pd.concat([training_examples, table], ignore_index=True) # Store the table

print_success('Finished extracting training examples:')
print(training_examples['label'].value_counts())
