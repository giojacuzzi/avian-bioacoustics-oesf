# Demonstrates the use of birdnetlib on a file, with and without source separation
from analyze import *

path = '/Users/giojacuzzi/Desktop/audio_test/chorus.wav'

# analyze (no separation)
print('\nanalyze---------------------------------------------')
detections_1 = analyze(path)
# print(*detections, sep = '\n')
print(get_max_conf_per_species(detections_1))

# analyze_with_separation 4
print('\nanalyze_with_separation 4---------------------------')
detections_4 = analyze_with_separation(path, 4)
print(get_max_conf_per_species(detections_4))

# analyze_with_separation 8
print('\nanalyze_with_separation 8---------------------------')
detections_8 = analyze_with_separation(path, 8)
print(get_max_conf_per_species(detections_8))

# TODO: break down predicitions for each time segment
