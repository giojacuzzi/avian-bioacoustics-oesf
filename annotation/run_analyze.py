# Run birdnetlib on a file with source separation and a species list
from analyze import *
import separate
import subprocess

# path = '/Users/giojacuzzi/Downloads/test.wav'
path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/Annotation/Data/SMA00404_20230518/SMA00404_20230518_052101_SS.wav'

def get_max_conf_per_species(detections):
    # TODO: THIS IS INCORRECT!
    return(detections.groupby('common_name').agg({'scientific_name': 'first', 'confidence': 'max', 'start_time': 'first', 'file': 'first'}).reset_index().sort_values(by='confidence', ascending=False))

if os.path.exists(separate.get_output_path()):
    shutil.rmtree(os.path.dirname(separate.get_output_path()))

print('\nanalyze (no separation)-----------------------------')
detections_1 = analyze(path)
print(get_max_conf_per_species(detections_1))

# analyze_with_separation 8
print('\nanalyze_with_separation 8---------------------------')
detections_8 = analyze_with_separation(path, 8, cleanup=False)
print(detections_8)
print(get_max_conf_per_species(detections_8))

# Open folder in finder
# subprocess.run(['/usr/bin/open', separate.get_output_path()])