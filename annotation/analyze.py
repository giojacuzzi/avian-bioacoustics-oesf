from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
import pandas as pd
import os
from subprocess import *
import shutil
import separate

# Load and initialize BirdNET-Analyzer model
species_list_path = os.path.abspath('species_list.txt')
print(f'Initializing BirdNET model with species list {species_list_path}...')
analyzer = Analyzer(custom_species_list_path=species_list_path)

# path - path to a .wav file
# df - return as a data frame (if not, raw list)
def analyze(path, df=True):
    recording = Recording(analyzer, path, min_conf=0.1)
    print(f'analyzing {path}...')
    recording.analyze()
    detections = recording.detections
    for d in detections:
        d['file'] = os.path.basename(path)
    if df:
        detections = pd.DataFrame(detections).sort_values('start_time')
    return(detections)

# 'cleanup' will remove any temporary files
def analyze_with_separation(path, num_sources = 4, cleanup = True):
    
    files = separate.separate(path, num_sources)

    # Aggregate detections
    detections = []
    for f in files:
        f_detections = analyze(f, df=False)
        if (len(f_detections) > 0):
            for d in f_detections:
                d['file'] = os.path.basename(f)
            detections = detections + f_detections
        else:
            print('no detections')
    
    if cleanup:
        shutil.rmtree(os.path.dirname(files[0]))

    # Find indices of maximum confidence values for each unique start_time and common_name combination
    df_detections = pd.DataFrame(detections).sort_values('start_time')
    idx = df_detections.groupby(['start_time', 'common_name'])['confidence'].idxmax()
    df_detections = df_detections.loc[idx]
    return df_detections
    
