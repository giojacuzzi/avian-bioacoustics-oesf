## Given an original file, a start time, and an end time, create a new file that extracts that segment of audio from the original

from datetime import datetime
import pandas as pd
from pydub import AudioSegment
from pydub.playback import play
from helper import *

# TODO: Get file from data
print('TEST')
csv_file_in = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/top_detections/2020/Deployment1/S4A04271_20200412_Data.csv'
path_out = 'TODO'

# DEBUG
# df = pd.read_csv(csv_file_in)
common_name = 'Barred Owl'
confidence = 0.931456
start_date = '2020-04-18 23:19:31'
end_date = '2020-04-18 23:19:34'
file = 'S4A04271_20200418_225943' + '.wav'

root_dir = '/Volumes/gioj_b1/OESF'

# Find the file matching 'file' under 'root_dir'
print('Finding file path...')
file_path = find_file_full_path(root_dir, file)
print(file_path)

## Given an audio file and the start/end times for a detection event, create a new audio file that extracts the detection event
# file_in - path to the original file that contains the detection event audio data
# path_out - path to the output file that will be created
# date_detection_start, date_detection_end - datetime objects
# buffer - seconds of audio to include before and after the detection
def extract_detection_audio(file_in, path_out, date_detection_start, date_detection_end, tag='', buffer=1.0):

    info = get_info_from_filename(file_in)
    date_file = datetime.datetime(int(info['year']), int(info['month']), int(info['day']), int(info['hour']), int(info['min']), int(info['sec']))

    file_out = f"{info['serial_no']}_{date_detection_start.strftime('%Y%m%d_%H%M%S')}_D-{tag}"
    print(f'Extracting detection to {file_out}...')

    # Load the entire audio file
    audio = AudioSegment.from_wav(file_in)
    audio = remove_dc_offset(audio)

    # Get start and end time relative to original file using datetime deltas
    file_start_time_ms  = int(date_file.timestamp() * 1000)
    file_end_time_ms  = file_start_time_ms + len(audio)
    detection_start_time_ms = max(file_start_time_ms, int(date_detection_start.timestamp() * 1000) - buffer * 1000)
    detection_end_time_ms   = min(file_end_time_ms, int(date_detection_end.timestamp() * 1000) + buffer * 1000)

    # Calculate the start and end positions for the specific detection
    start_position = detection_start_time_ms - file_start_time_ms
    end_position = start_position + (detection_end_time_ms - detection_start_time_ms)

    # Extract the specific detection segment
    detection_segment = audio[start_position:end_position]
    print(len(detection_segment))
    print(detection_end_time_ms - detection_start_time_ms)

    # TODO: Save the extracted file
    # detection_segment.export(path_out, format="wav")
    play(detection_segment)

date_format = '%Y-%m-%d %H:%M:%S'
date_detection_start = datetime.datetime.strptime(start_date, date_format)
date_detection_end   = datetime.datetime.strptime(end_date, date_format)
extract_detection_audio(file_path, path_out, date_detection_start, date_detection_end, tag=f'{common_name}-{confidence}')

print('TEST DONE')