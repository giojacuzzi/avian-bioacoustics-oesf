from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
import pandas as pd

# Load and initialize BirdNET-Analyzer models
analyzer = Analyzer()

recording = Recording(
    analyzer,
    '/Users/giojacuzzi/Desktop/oesf-examples/short/04_short.wav',
    lat=47.676786,
    lon=-124.136721,
    date=datetime(year=2020, month=7, day=8), # use date or week_48
    min_conf=0.05,
)
recording.analyze()

# Find max confidence per species
df = pd.DataFrame(recording.detections)
df = df.groupby('common_name')['confidence'].max().reset_index().sort_values(by='confidence', ascending=False)
print(df)
