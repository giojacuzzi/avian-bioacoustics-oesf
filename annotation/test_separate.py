import separate

path = '/Users/giojacuzzi/Desktop/audio_test/chorus.wav'
num_sources = 4

# Multichannel file
file = separate.separate(path, num_sources, multichannel=True)
print(file)

# Individual files
files = separate.separate(path, num_sources)
print(files)
