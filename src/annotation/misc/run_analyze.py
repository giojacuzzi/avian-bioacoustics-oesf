# Run birdnetlib on a file with source separation and a species list
from analyze import *
import classification.sound_separation as sound_separation
import subprocess
import pandas as pd

path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/SMA00351_20200412/SMA00351_T060036_D20200416.wav'

# def get_max_conf_per_species(detections):
#     if (len(detections) > 0):
#         # TODO: THIS IS INCORRECT!
#         return(detections.groupby('common_name').agg({'scientific_name': 'first', 'confidence': 'max', 'start_time': 'first', 'file': 'first'}).reset_index().sort_values(by='confidence', ascending=False))
#     else:
#         return([])

if os.path.exists(sound_separation.get_output_path()):
    shutil.rmtree(os.path.dirname(sound_separation.get_output_path()))

pd.set_option('display.max_rows', None)

print('\nanalyze (no separation)-----------------------------')
detections_1 = analyze(path)
print(*detections_1, sep = '\n')
# print(get_max_conf_per_species(detections_1))

# analyze_with_separation 8
print('\nanalyze_with_separation 8---------------------------')
detections_8 = analyze_with_separation(path, 8, cleanup=False)
print(*detections_8, sep = '\n')
# print(get_max_conf_per_species(detections_8))

# Open folder in finder
# subprocess.run(['/usr/bin/open', separate.get_output_path()])