# NOTE: Custom edits to birdnetlib required (see analyzer.py).
# Specifically, APPLY_SIGMOID flag set to False throughout to
# return logits, not sigmoid activations
from birdnetlib.analyzer import Analyzer, MODEL_VERSION

import datetime
import os
import pandas as pd
import time
from . import analyze
from utils.log import *
from utils.files import *
from utils.bnl import *
import sys

# Create a global analyzer instance
if 'analyzer' not in locals() and 'analyzer' not in globals():

    # Check for correct version for this project
    expected_version = '2.4'
    if MODEL_VERSION != expected_version:
        print_error(f'birdnetlib.analyzer MODEL_VERSION {MODEL_VERSION} incompatible. Please install the expected version {expected_version}')
        sys.exit()

    species_list_path = os.path.abspath('src/classification/species_list/species_list_OESF.txt')
    print(f'Initializing analyzer with species list {species_list_path}')
    analyzer = Analyzer(custom_species_list_path=species_list_path)

# Run the analyzer on the given file and save the resulting detections to a csv
def process_file(
        in_filepath,
        out_dir        = '',
        root_dir       = None,
        min_confidence = 0.0,
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

    already_analyzed = list_base_files_by_extension(out_dir, 'csv')
    already_analyzed = [f.rstrip('.csv') for f in already_analyzed]
    if (os.path.splitext(os.path.basename(in_filepath))[0]) in already_analyzed:
        print(f'  {os.path.basename(in_filepath)} already analyzed. SKIPPING...')
        return

    # Run analyzer to obtain detections
    try:
        info = parse_metadata_from_filename(in_filepath)

        start_time_file = time.time()
        result = analyze.analyze_detections(
            filepath = in_filepath,
            analyzer = analyzer,
            min_confidence = min_confidence,
            num_separation = num_separation,
            cleanup = cleanup
        )

        if info is None:
            dt = datetime.timedelta(0)
        else:
            dt = datetime.datetime(int(info['year']), int(info['month']), int(info['day']), int(info['hour']), int(info['min']), int(info['sec']))

        col_names = ['common_name','confidence','logit','start_date']
        if not result.empty:
            start_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['start_time'])))
            start_dates = list(map(lambda d: dt + d, start_deltas))

            # Create columns for raw logit value, rounded sigmoid activated confidence, and start date
            result = result.rename(columns={'confidence': 'logit'})
            result['confidence'] = sigmoid_BirdNET(result['logit'])
            result['start_date'] = start_dates

            # end_deltas = list(map(lambda s: datetime.timedelta(days=0, seconds=s), list(result['end_time'])))
            # end_dates = list(map(lambda d: dt + d, end_deltas))
            # result['end_date'] = end_dates

            result = result[col_names] # only keep essential values
        else:
            result = pd.DataFrame(columns=col_names)
        
        # Discard any detections below the minimum confidence and sort results
        result = result[result['confidence'] >= min_confidence]
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
        print_error(f'{str(e)}\n{in_filepath}')
