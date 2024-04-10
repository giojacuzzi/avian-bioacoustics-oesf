import os                       # File navigation
import pandas as pd             # Data manipulation
from utils import files
from utils.log import *

# Returns a dataframe of raw metadata from all raw acoustic monitoring data under the given directories
def get_raw_metadata(dirs=[], overwrite=False):

    metadata_filepath = 'data/metadata/processed/raw_metadata.csv'

    # Return previously processed annotations unless overwrite requested
    if os.path.exists(metadata_filepath) and not overwrite:
        print(f'Retrieving metadata from {metadata_filepath}')
        return pd.read_csv(metadata_filepath, index_col=None)
    
    if (overwrite and len(dirs) == 0) or (not overwrite and not os.path.exists(metadata_filepath) and len(dirs) == 0):
        print_error('At least one directory must be provided to write metadata to file')
        return
    
    print('Retrieving metadata from raw data...')

    # Declare 'evaluated_detections' - Get the list of files (detections) that were actually evaluated by annotators
    # These are all the .wav files under each site-date folder in e.g. 'annotation/data/_annotator/2020/Deployment4/SMA00410_20200523'
    filepaths = []
    for dir in dirs:
        filepaths.extend(files.find_files(dir, '.wav'))

    if len(filepaths) == 0:
        print_error('No .wav files found')
        exit()
    else:
        print(f'Found {len(filepaths)} .wav files')

    # Parse metadata from the .wav filenames
    file_metadata = list(map(lambda f: files.parse_metadata_from_raw_audio_filepath(f), filepaths))
    deployments, serialnos, dates, times = zip(*file_metadata)

    file_metadata = pd.DataFrame({
        'filepath':   filepaths,
        'deployment': deployments,
        'date':       dates,
        'time':       times,
        'serialno':   serialnos
    })
    file_metadata = file_metadata.sort_values(by=['deployment','serialno','date', 'time'])

    file_metadata.to_csv(metadata_filepath, index=False)
    print_success(f'Saved annotations to {metadata_filepath}')
    return file_metadata

# Run the routine
raw_metadata = get_raw_metadata(
    [
        '/Volumes/gioj_b1/OESF/2020'
    ],
    overwrite=False)

print(raw_metadata)