from astral import LocationInfo
from astral.location import Location
from astral.sun import sun
from datetime import date, datetime, timedelta
import pandas as pd
import os
import re
import numpy as np
from random import shuffle
import sys

# Regex to match filename convention
filename_convention = r'(?P<id>[^_]*)_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2}).*'

def get_metadata_from_filename(path):
    filename = os.path.basename(path)
    pattern = re.compile(filename_convention)
    match = pattern.match(filename)
    if not match:
        raise ValueError(f"The filename '{filename}' does not match the expected pattern.")

    metadata = match.groupdict()
    return metadata

def find_wav_files(directory):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                wav_path = os.path.join(root, file)
                wav_files.append(wav_path)
    return wav_files

def find_files_by_date(directory, year, month, day):
    pattern = re.compile(filename_convention)
    all_files = os.listdir(directory)
    # print(all_files)

    # Filter files based on the filename components
    matching_files = []
    for filename in all_files:
        # print(filename)
        match = pattern.match(filename)
        if match:
            components = match.groupdict()
            if (
                int(components['year']) == year and
                int(components['month']) == month and
                int(components['day']) == day
            ):
                matching_files.append(os.path.join(directory, filename))

    return sorted(matching_files)

def get_start_sunrise_sunset_end(id, lat, lon, year, month, day):
    locale = LocationInfo(id, 'OESF', "America/Los_Angeles", lat, lon)
    print(Location(locale).timezone)
    print((f"{locale.name} ({locale.latitude},{locale.longitude})\n"))
    s = sun(locale.observer, date=date(year, month, day), tzinfo=Location(locale).timezone)
    t_sunrise = pd.to_datetime(s["sunrise"]).round('1s')
    t_sunset  = pd.to_datetime(s["sunset"]).round('1s')
    t_24h_start = t_sunrise.normalize()
    t_24h_end = t_24h_start + timedelta(days=1)
    print((
        f'Start: {t_24h_start}\n'
        f'Sunrise: {t_sunrise}\n'
        f'Sunset:  {t_sunset}\n'
        f'End:  {t_24h_end}\n'
    ))
    return t_24h_start, t_sunrise, t_sunset, t_24h_end

# TESTING DEBUG
# YOU MUST SPECIFY THIS
id    = 'SMA00556'
lat   = 47.67369
lon   = -124.33503
year  = 2023
month_start = 6
month_end   = 6
day_start   = 12
day_end     = 21
path  = f'/Volumes/gioj/OESF/2023/D3_20230611_20230620/SMA00556_20230611_000000'

# DON'T TOUCH BELOW

# get deployment start, end, and middle dates
start_datetime = datetime(year=year, month=month_start, day=day_start)
end_datetime = datetime(year=year, month=month_end, day=day_end) + timedelta(days=1) - timedelta(seconds=1)
days_difference = (end_datetime - start_datetime).days
middle_date = start_datetime + timedelta(days=days_difference // 2)

# get sunrise, sunset, and start/end of day for middle date
t_24h_start, t_sunrise, t_sunset, t_24h_end = get_start_sunrise_sunset_end(id, lat, lon, middle_date.year, middle_date.month, middle_date.day)

subsample_len = 12 # seconds per subsample
subsample_freq_dense = '5min' # frequency of subsamples in dense periods
subsample_freq_sparse = '15min' # frequency of subsamples in sparse periods

# define sampling period intervals
periods = [
    # Sparse from midnight to 1h before sunrise
    pd.date_range(start=(t_24h_start), end=(t_sunrise - timedelta(hours=1)), freq=subsample_freq_sparse, inclusive='left'),
    # Dense from 1h before sunrise to 4h after
    pd.date_range(start=(t_sunrise - timedelta(hours=1)), end=(t_sunrise + timedelta(hours=4)), freq=subsample_freq_dense, inclusive='left'),
    # Sparse from 4h after sunrise to 4h before sunset
    pd.date_range(start=(t_sunrise + timedelta(hours=4)), end=(t_sunset - timedelta(hours=4)), freq=subsample_freq_sparse, inclusive='left'),
    # Dense from 4h before sunset to sunset
    pd.date_range(start=(t_sunset - timedelta(hours=4)), end=(t_sunset), freq=subsample_freq_dense, inclusive='left'),
    # Sparse from sunset to midnight
    pd.date_range(start=(t_sunset), end=(t_24h_end), freq=subsample_freq_sparse, inclusive='left')
]

# shuffle times
timedeltas = pd.TimedeltaIndex([])
for p in periods:
    timedeltas = timedeltas.append(pd.to_timedelta(p.strftime('%H:%M:%S')))
timedeltas_shuffled = timedeltas.to_list()
shuffle(timedeltas_shuffled)

# shuffle an (approximately) equal number of dates for each time
dates = pd.date_range(start=start_datetime, end=end_datetime)
n = len(timedeltas)
dates = (dates.tolist() * (n // len(dates))) + dates.tolist()[:n % len(dates)]
dates_shuffled = dates
shuffle(dates_shuffled)

# combine shuffled dates and times
results = []
for i in range(0, len(timedeltas)):
    results.append(dates_shuffled[i].normalize().to_pydatetime() + timedeltas_shuffled[i])

# print('FINAL')
# for r in results:
#     print(r)
return results