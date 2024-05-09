## Extract annotated audio examples as training data for a custom model

from utils.files import *
from utils.audio import *
import string
import os
import sys
from pydub import AudioSegment

# Path to directory that contains the annotated Raven Pro selection tables
dir_training_input_annotations = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/training_data'
# Path to directory that will be populated with training data
dir_training_output_data = '/Users/giojacuzzi/Desktop/OESF_training_data'
# Path to directory with raw audio data
data_dir = '/Volumes/gioj_b1/OESF'

# Given a list of classes to train...
species_classes = ['red-breasted nuthatch'] # species classes
classes_to_train = species_classes + [] # e.g. 'background' or others

species_list = pd.read_csv(species_list_filepath, index_col=None, usecols=['common_name', 'scientific_name'])
species_list['common_name']     = species_list['common_name'].str.lower()
species_list['scientific_name'] = species_list['scientific_name'].str.lower()

training_examples = pd.DataFrame()

# For each class...
for common_name in classes_to_train:

    # Format the class name to match BirdNET
    print(f'Processing class "{common_name}"')
    row = species_list.loc[species_list['common_name'] == common_name]
    if not row.empty:
        scientific_name = row.iloc[0]['scientific_name']
        class_label = f"{scientific_name.capitalize()}_{string.capwords(common_name)}"
        print(class_label)
    else:
        print_error(f"Scientific name for {common_name} not found.")
        continue

    # Create the output directory
    class_output_dir = dir_training_output_data + '/' + class_label
    if not os.path.exists(class_output_dir):
        os.makedirs(class_output_dir)

    # Get the associated annotations for the class
    dir_class_input_annotations = dir_training_input_annotations + '/' + common_name
    annotation_files = find_files(dir_class_input_annotations, 'selections.txt')
    if len(annotation_files) == 0:
        print_error(f'No files found for class "{common_name}"')
        continue

    # Use the annotations to extract examples from the raw audio data
    for file in annotation_files:

        print(f'Loading annotation file {file}...')

        # Parse metadata from file
        metadata = parse_metadata_from_filename(file)
        print(metadata)
        
        # Load the annotations
        table = load_raven_selection_table(file, cols_needed=['selection', 'class', 'type', 'begin file', 'file offset (s)', 'delta time (s)', 'low freq (hz)', 'high freq (hz)'])

        # Remove annotation references (these are used in the test dataset)
        table = table[(table['low freq (hz)'] != 0.0) & (table['high freq (hz)'] != 16000.0)]

        # Further validate the annotations
        table_errors = table[table['delta time (s)'] > 3.0]
        for index, row in table_errors.iterrows():
            print_error(f"Selection {row['selection']} extends beyond 3 seconds in {file}")

        # Keep only annotations for the class
        annotations = table[table['class'] == common_name]
        # print(annotations)

        # Extract the raw audio data to the training data directory
        # print(dir_training_output_data)
        for index, row in annotations.iterrows():
            print(f'Extracting audio data for row {row}')

            # Find the original audio file matching 'file' under 'root_dir'
            audio_file = find_file_full_path(data_dir, row['begin file'])
            if audio_file is None:
                print_error("Could note find corresponding .wav file")
                continue

            # Load the entire audio file
            audio_data = AudioSegment.from_wav(audio_file)
            audio_data = remove_dc_offset(audio_data)

            annotation_start_time = row['file offset (s)']
            annotation_end_time = annotation_start_time + row['delta time (s)']
            midpoint = annotation_start_time + (annotation_end_time - annotation_start_time)/2.0

            # Create a 3-second window centered at the midpoint of the annotation
            data_start_time = midpoint - 3.0/2.0
            data_end_time = data_start_time + 3.0

            # Extract the data and save to file
            extracted_data = audio_data[(data_start_time * 1000):(data_end_time * 1000)]
            file_out = f"{metadata['serial_no']}_{metadata['date']}_{metadata['time']}_{data_start_time}.wav"
            path_out = f'{class_output_dir}/{file_out}'
            print(f'Saving to {path_out}')
            extracted_data.export(path_out, format='wav')

        # Store info for later summary
        training_examples = pd.concat([training_examples, annotations], ignore_index=True) # Store the table

print_success('Finished extracting training examples:')
print(training_examples['class'].value_counts())
