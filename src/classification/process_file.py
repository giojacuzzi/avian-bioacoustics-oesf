from birdnetlib.analyzer import Analyzer, MODEL_VERSION
import birdnetlib.analyzer
from birdnetlib import Recording 
import os
import pandas as pd
from . import sound_separation
from subprocess import *
import shutil
import datetime
import time
from utils.log import *
from utils.files import *
from utils.bnl import *
import sys

# # TODO: cache analyzer
# # Create a global analyzer instance
# if 'analyzer' not in locals() and 'analyzer' not in globals():

#     # Check for correct version for this project
#     expected_version = '2.4-Gio'
#     if MODEL_VERSION != expected_version:
#         print_error(f'birdnetlib.analyzer MODEL_VERSION {MODEL_VERSION} incompatible. Please install the expected version {expected_version}')
#         sys.exit()

#     species_list_path = os.path.abspath('src/classification/species_list/species_list_OESF.txt')
#     print(f'Initializing analyzer with species list {species_list_path}')
#     # analyzer = Analyzer(custom_species_list_path=species_list_path)
#     analyzer = Analyzer(classifier_model_path='/Users/giojacuzzi/Desktop/OESF_training_output/Custom_Classifier.tflite', classifier_labels_path='/Users/giojacuzzi/Desktop/OESF_training_output/Custom_Classifier_Labels.txt')

# path - path to a .wav file
# cleanup - remove any temporary files created during analysis
# num_separation - the number of sound channels to separate. '1' will leave the original file unaltered.
def analyze_detections(filepath, analyzer, min_confidence, apply_sigmoid=True, num_separation=1, cleanup=True):
    
    if (num_separation > 1):
        files = sound_separation.separate(filepath, num_separation)
    else:
        files = [filepath] # original file, no sound separation

    # Obtain detections across file(s)
    detections = pd.DataFrame()
    for file in files:

        birdnetlib.analyzer.APPLY_SIGMOID = apply_sigmoid # Overwrite flag to return logit values instead of sigmoid transforms
        
        recording = Recording(
            analyzer=analyzer,
            path=file,
            min_conf=min_confidence,
        )
        recording.minimum_confidence = min_confidence # necessary override to enforce 0.0 case
        recording.analyze()
        file_detections = pd.DataFrame(recording.detections)
        print(f'analyze_detections found {str(len(file_detections))} detections')

        file_detections['file_origin'] = os.path.basename(file)
        if (len(file_detections) > 0):
            detections = pd.concat([detections, file_detections], ignore_index=True)
        else:
            print('no detections')
    
    if num_separation == 1:
        return detections
    else:
        # Aggregate detections for source separation
        if cleanup:
            shutil.rmtree(os.path.dirname(files[0]))

        # Find indices of maximum confidence values for each unique start_time and common_name combination
        aggregated_detections = pd.DataFrame(detections).sort_values('start_time')
        i = aggregated_detections.groupby(['start_time', 'common_name'])['confidence'].idxmax()
        aggregated_detections = aggregated_detections.loc[i]
        return aggregated_detections

# Run the analyzer on the given file and save the resulting detections to a csv
def process_file(
        in_filepath,
        out_dir        = '',
        root_dir       = None,
        analyzer_filepath = None,
        labels_filepath = 'src/classification/species_list/species_list_OESF.txt',
        min_confidence = 0.0,
        apply_sigmoid  = True,
        num_separation = 1,
        cleanup        = True,
        sort_by        = 'start_date',
        ascending      = True,
        save_to_file   = True
):
    # Save directly to output directory
    if root_dir is None:
        file_out = os.path.basename(os.path.splitext(in_filepath)[0]) + '.csv'
    # Save to the output directory, preserving the original directory structure relative to the root
    else:
        if not root_dir in in_filepath:
            print_error('Root directory must contain input file path')
            return
        file_out = os.path.splitext(in_filepath[len(root_dir):])[0] + '.csv'
    path_out = os.path.normpath(out_dir + '/' + file_out)

    if analyzer_filepath is None:
        # Use default pre-trained analyzer
        labels_filepath = os.path.abspath(labels_filepath)
        print(f'Initializing pre-trained analyzer with species list {labels_filepath}')
        analyzer = Analyzer(custom_species_list_path=labels_filepath)
        # TODO: cache analyzer
    else:
        analyzer_filepath = os.path.abspath(analyzer_filepath)
        labels_filepath = os.path.abspath(labels_filepath)
        print(f'Initializing custom analyzer {analyzer_filepath} with species list {labels_filepath}')
        analyzer = Analyzer(classifier_model_path=analyzer_filepath, classifier_labels_path=labels_filepath)
        # TODO: cache analyzer

    already_analyzed = list_base_files_by_extension(out_dir, 'csv')
    already_analyzed = [f.rstrip('.csv') for f in already_analyzed]
    if (os.path.splitext(os.path.basename(in_filepath))[0]) in already_analyzed:
        print(f'  {os.path.basename(in_filepath)} already analyzed. SKIPPING...')
        return

    # Run analyzer to obtain detections
    try:
        info = parse_metadata_from_filename(in_filepath)

        start_time_file = time.time()
        result = analyze_detections(
            filepath = in_filepath,
            analyzer = analyzer,
            min_confidence = min_confidence,
            apply_sigmoid = apply_sigmoid,
            num_separation = num_separation,
            cleanup = cleanup
        )

        if info is None:
            dt = datetime.timedelta(0)
        else:
            dt = datetime.datetime(int(info['year']), int(info['month']), int(info['day']), int(info['hour']), int(info['min']), int(info['sec']))

        # Only keep essential data
        if apply_sigmoid:
            col_names = ['common_name','confidence','start_date']
        else:
            col_names = ['common_name','confidence','logit','start_date']

        if not result.empty:

            # Create columns for start date and (if applicable) raw logit value and rounded sigmoid activated confidence score
            start_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['start_time'])))
            start_dates = list(map(lambda d: dt + d, start_deltas))
            result['start_date'] = start_dates

            # end_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['end_time'])))
            # end_dates = list(map(lambda d: dt + d, end_deltas))
            # result['end_date'] = end_dates

            # If analyzer returned raw logit values in the 'confidence' column, create
            # a unique column each for logit values and sigmoid confidence scores.
            if not apply_sigmoid:
                result = result.rename(columns={'confidence': 'logit'})
                result['confidence'] = sigmoid_BirdNET(result['logit'])

            result = result[col_names]

        else:
            result = pd.DataFrame(columns=col_names)

        result = result.sort_values(sort_by, ascending=ascending)

        # Save results to file
        if save_to_file:
            if not os.path.exists(os.path.dirname(path_out)):
                os.makedirs(os.path.dirname(path_out))
            pd.DataFrame.to_csv(result, path_out, index=False) 

        end_time_file = time.time()
        print_success(f'Finished processing file {in_filepath}\n({(end_time_file - start_time_file):.2f} sec)')

        return result

    except Exception as e:
        print_error(f'({type(e).__name__}) {e}\n{in_filepath}')

##### TEST
# species_list_path = os.path.abspath('classification/species_list/species_list_OESF.txt')
# print(analyze_detections(
#     filepath = '/Users/giojacuzzi/Desktop/audio_test/1/SMA00351_20200414_060036.wav',
#     analyzer = Analyzer(custom_species_list_path=species_list_path),
#     min_confidence = 0.5,
#     num_separation = 1,
#     cleanup = True
# ))
# print(analyze_detections(
#     filepath = '/Users/giojacuzzi/Desktop/audio_test/1/SMA00351_20200414_060036.wav',
#     analyzer = Analyzer(custom_species_list_path=species_list_path),
#     min_confidence = 0.5,
#     num_separation = 4,
#     cleanup = True
# ))
