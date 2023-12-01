from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
import pandas as pd
import os
from subprocess import *
import shutil
import separate

# Load and initialize BirdNET-Analyzer model
print('Initializing BirdNET model...')
analyzer = Analyzer()

# returns a pandas dataframe
def get_max_conf_per_species(detections):
    return(pd.DataFrame(detections).groupby('common_name')['confidence'].max().reset_index().sort_values(by='confidence', ascending=False))

# path - path to a .wav file
def analyze(path):
    recording = Recording(
        analyzer, path,
        # TODO: use species list
        min_conf=0.1,
    )
    print(f'analyzing {path}...')
    recording.analyze()
    return(recording.detections)

# 'cleanup' will remove any temporary files
def analyze_with_separation(path, num_sources = 4, cleanup = True):
    
    files = separate.separate(path, num_sources)

    detections = []
    for f in files:
        f_detections = analyze(f)
        if (len(f_detections) > 0):
            # print(get_max_conf_per_species(f_detections))
            detections = detections + f_detections
        else:
            print('no detections')

    if cleanup:
        shutil.rmtree(os.path.dirname(files[0]))

    return detections
    
