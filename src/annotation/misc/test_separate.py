# Find results in annotation/_output/temp

import classification.separate as separate

path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/SMA00351_20200412/SMA00351_20200414_060036.wav'
num_sources = 4

# Multichannel file
file = separate.separate(path, num_sources, multichannel=True)
print(file)

# Individual files
files = separate.separate(path, num_sources)
print(files)
