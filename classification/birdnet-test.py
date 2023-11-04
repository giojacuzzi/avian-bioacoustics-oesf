from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from birdnetlib import species
import pandas as pd

species_list = species.getSpeciesList(lat=47.676786, lon=-124.136721, week=-1, threshold=0.05)

""" # Load and initialize BirdNET-Analyzer models
analyzer = Analyzer()

recording = Recording(
    analyzer,
    '/Users/giojacuzzi/Desktop/Kitchen Sink/oesf-examples/2020-04-19/RX/Selection_RX_MATURE_SMA00393_20200419_dawn.wav',
    #lat=47.676786,
    #lon=-124.136721,
    #date=datetime(year=2020, month=7, day=8), # use date or week_48
    min_conf=0.05,
)
recording.analyze()

# Find max confidence per species
df = pd.DataFrame(recording.detections)
df = df.groupby('common_name')['confidence'].max().reset_index().sort_values(by='confidence', ascending=False)
print(df) """
