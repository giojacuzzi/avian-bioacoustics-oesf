from .log import *
import os
import re
import pandas as pd
from utils.log import *

# File with table containing ID and ecological data for monitored species
species_list_filepath = 'data/species/species_list - complete.csv'

#
def find_file_full_path(top_directory, filename):
    for root, dirs, files in os.walk(top_directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Function to parse species name, confidence, serial number, date, and timecfrom annotation filename,
# e.g. "Yellow-rumped Warbler-0.4231688380241394_SMA00399_20200628_204241.wav"
def parse_metadata_from_annotation_file(filename):

    # Regular expression pattern to match the filename
    pattern = r'^(.+)-([\d.]+)_(\w+)_(\d{8})_(\d{6})\.(\w+)$'
    match = re.match(pattern, filename)

    if match:
        species    = match.group(1)
        confidence = float(match.group(2))
        serialno   = match.group(3)
        date       = match.group(4)
        time       = match.group(5)
        return species, confidence, serialno, date, time
    else:
        print_warning("Unable to parse info from filename:", filename)

# Function to parse species name, confidence, serial number, date, and time from raw audio filepath,
# e.g. ".../Deployment6/SMA00399_20200619/SMA00399_20200619_040001.wav"
def parse_metadata_from_raw_audio_filepath(filepath):

    # Regular expression pattern to match the deployment number
    deployment = re.findall(".Deployment([\d.]+)", filepath)
    if (len(deployment) != 1):
        print_error(f'Could not parse Deployment token from {filepath}')
        return
    else:
        deployment = deployment[0]

    # Regular expression pattern to match the filename
    pattern = r'(?P<serialno>\S+?)_(?P<date>\d{8}?)_(?P<time>\d{6}?)\.(?P<type>\w+?)$'
    match = re.search(pattern, os.path.basename(filepath))
    if match:
        serialno = match.group('serialno')
        date     = match.group('date')
        time     = match.group('time')
        return deployment, serialno, date, time
    else:
        print_error(f'Unable to parse info from filepath: {filepath}')

# Find all selection table files under a root directory
def find_files(directory, suffix=None, prefix=None):

    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            suffix_match = (suffix is None) or (suffix is not None and file.endswith(suffix))
            prefix_match = (prefix is None) or (prefix is not None and file.startswith(prefix))
            if suffix_match and prefix_match:
                results.append(os.path.join(root, file))
    return results

# Scrape serial number, date, and time from Song Meter filename
def parse_metadata_from_filename(path):
    filename = os.path.basename(path)
    substrs = filename.split('.')[0].split('_')
    if len(substrs) != 3:
        print_warning('Could not get metadata from filename')
        return
    date = substrs[1]
    time = substrs[2]
    return({
        'serial_no': substrs[0],
        'date':      date,
        'year':      date[0:4],
        'month':     date[4:6],
        'day':       date[6:8],
        'time':      time,
        'hour':      time[0:2],
        'min':       time[2:4],
        'sec':       time[4:6],
    })

def getDirectoriesWithFiles(path, filetype):
    directoryList = []
    if os.path.isfile(path):
        return []
    # Add dir to directorylist if it contains files of filetype
    if len([f for f in os.listdir(path) if f.endswith(filetype)]) > 0:
        directoryList.append(path)
    for d in os.listdir(path):
        new_path = os.path.join(path, d)
        if os.path.isdir(new_path):
            directoryList += getDirectoriesWithFiles(new_path, filetype)
    return directoryList

def list_base_files_by_extension(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(f'.{extension}'):
                file_list.append(filename)
    return file_list

def list_files_in_directory(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

# Returns a dataframe of raw metadata from all raw acoustic monitoring data under the given directories
def get_raw_metadata(dirs=[], overwrite=False):

    metadata_filepath = 'data/metadata/raw_metadata.csv'

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
        filepaths.extend(find_files(dir, suffix='.wav'))

    if len(filepaths) == 0:
        print_error('No .wav files found')
        exit()
    else:
        print(f'Found {len(filepaths)} .wav files')

    # Parse metadata from the .wav filenames
    file_metadata = list(map(lambda f: parse_metadata_from_raw_audio_filepath(f), filepaths))
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
