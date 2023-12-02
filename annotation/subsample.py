from astral import LocationInfo
from astral.location import Location
from astral.sun import sun
from datetime import date, datetime, timedelta
import pandas as pd
import os
import re

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

def subsample(id, lat, lon, year, month, day):
    # """
    # This function takes a path as input and prints information about the path.
    
    # Args:
    # - path (str): The path to be processed.
    # """
    # if not os.path.exists(path) or not os.path.isfile(path):
    #     raise FileNotFoundError(f"The file at '{path}' does not exist.")

    # print(f"Subsampling {os.path.abspath(path)}...")

    locale_date = date(year, month, day)

    locale = LocationInfo(id, 'OESF', "America/Los_Angeles", lat, lon)
    print((f"{locale.name} ({locale.latitude},{locale.longitude})\n"))
    s = sun(locale.observer, date=locale_date, tzinfo=Location(locale).timezone)
    t_sunrise = pd.to_datetime(s["sunrise"]).round('1s')
    t_sunset  = pd.to_datetime(s["sunset"]).round('1s')
    print((
        f'Sunrise: {t_sunrise}\n'
        f'Sunset:  {t_sunset}\n'
    ))

    # Sunrise: 4 hr block, beginning 1 hr before sunrise, @ 4 samples / hr (every 15 min)
    tp_sunrise = pd.date_range(start=(t_sunrise - timedelta(hours=1)), periods=(4 * 4), freq='15min')

    # Sunset: 2 hr block, beginning 1 hr before sunrise, @ 4 samples / hr (every 15 min)
    tp_sunset = pd.date_range(start=(t_sunset - timedelta(hours=1)), periods=(2 * 4), freq='15min')

    # Night (AM): @ 2 samples / hr (every 30 min)
    # Ending at 30 min before tp_sunrise[0]
    p = len(pd.date_range(start=t_sunrise.normalize(), end=(tp_sunrise[0]), freq='30min', inclusive='left')) - 1
    tp_night_am = pd.date_range(end=(tp_sunrise[0] - timedelta(minutes=30)), freq='30min', periods=p)

    # Day: @ 2 samples / hr (every 30 min)
    # Evenly-spaced between tp_sunrise[-1] + 15 min and tp_sunset[0] - 30 min
    # TODO!
    tp_day = pd.date_range(start=(tp_sunrise[-1] + timedelta(minutes=15)), end=(tp_sunset[0] - timedelta(minutes=15)), freq='30min')

    # Night (PM): @ 2 samples / hr (every 30 min)
    # Starting at 15 min after tp_sunset[-1] and ending by 23:59:00
    tp_night_pm = pd.date_range(start=(tp_sunset[-1] + timedelta(minutes=15)), end=(t_sunrise.normalize() + timedelta(days=1) - timedelta(minutes=1)), freq='30min')

    # Print time periods
    print(f'Night (AM): {len(tp_night_am)} samples')
    print([t.strftime('%H:%M:%S') for t in tp_night_am.to_pydatetime()])

    print(f'Sunrise: {len(tp_sunrise)} samples')
    print([t.strftime('%H:%M:%S') for t in tp_sunrise.to_pydatetime()])

    print(f'Day: {len(tp_day)} samples')
    print([t.strftime('%H:%M:%S') for t in tp_day.to_pydatetime()])

    print(f'Sunset: {len(tp_sunset)} samples')
    print([t.strftime('%H:%M:%S') for t in tp_sunset.to_pydatetime()])

    print(f'Night (PM): {len(tp_night_pm)} samples')
    print([t.strftime('%H:%M:%S') for t in tp_night_pm.to_pydatetime()])

    # Check total time periods sum to 60
    total = sum(len(tp) for tp in [tp_night_am, tp_sunrise, tp_day, tp_sunset, tp_night_pm])
    print('Total length:', total)

    tp_all = []
    for tp in [tp_night_am, tp_sunrise, tp_day, tp_sunset, tp_night_pm]:
        tp_all.extend(tp.time)

    # print(tp_all)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # Create a DataFrame with DatetimeIndex and data values
    df = pd.DataFrame({'Date': tp_night_am.union(tp_sunrise).union(tp_day).union(tp_sunset).union(tp_night_pm), 'Value': 1})
    df.set_index('Date', inplace=True)

    # # Plot
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df.index, df['Value'], s=100, color='blue', marker='o', edgecolors='black')
    # plt.title(f'{total} seconds')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.grid(True)
    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=12))  # Adjust the number of ticks as needed
    # plt.axvline(x=t_sunrise, color='red', linestyle='--')
    # plt.axvline(x=t_sunset, color='red', linestyle='--')
    # plt.show()

    return(tp_all)


