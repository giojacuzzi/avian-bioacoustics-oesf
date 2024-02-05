import datetime
import os
import pandas as pd
import time
import numpy as np

# Sigmoid activation for raw logit confidence values
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#
def find_file_full_path(top_directory, filename):
    for root, dirs, files in os.walk(top_directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Scrape serial number, date, and time from Song Meter filename
def get_info_from_filename(path):
    filename = os.path.basename(path)
    substrs = filename.split('.')[0].split('_')
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

import numpy as np
from pydub import AudioSegment

def remove_dc_offset(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    samples = samples - round(np.mean(samples))
    return(AudioSegment(samples.tobytes(), channels=audio_segment.channels, sample_width=audio_segment.sample_width, frame_rate=audio_segment.frame_rate))
