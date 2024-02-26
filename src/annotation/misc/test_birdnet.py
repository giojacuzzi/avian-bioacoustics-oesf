from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
import pandas as pd

# Load and initialize BirdNET-Analyzer model
print('Initializing BirdNET model...')
analyzer = Analyzer()

print('Analyzing file...')
recording = Recording(
    analyzer,
    '/Users/giojacuzzi/Desktop/audio_test/owl.wav',
    #lat=47.676786,
    #lon=-124.136721,
    #date=datetime(year=2020, month=7, day=8), # use date or week_48
    min_conf=0.05,
)
recording.analyze()

print(*recording.detections, sep = '\n')

# Find max confidence per species
df = pd.DataFrame(recording.detections)
df = df.groupby('common_name')['confidence'].max().reset_index().sort_values(by='confidence', ascending=False)
print(df)
