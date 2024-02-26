from birdnetlib import Recording 
import os
import pandas as pd
from . import sound_separation
from subprocess import *
import shutil

# path - path to a .wav file
# cleanup - remove any temporary files created during analysis
# num_separation - the number of sound channels to separate. '1' will leave the original file unaltered.
def analyze_detections(filepath, analyzer, min_confidence, num_separation=1, cleanup=True):
    
    if (num_separation > 1):
        files = sound_separation.separate(filepath, num_separation)
    else:
        files = [filepath] # original file, no sound separation

    # Obtain detections across file(s)
    detections = pd.DataFrame()
    for file in files:
        
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