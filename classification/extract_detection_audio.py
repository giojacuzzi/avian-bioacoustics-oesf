## Given an original file, a start time, and an end time, create a new file that extracts that segment of audio from the original

from datetime import datetime
import pandas as pd
from pydub import AudioSegment
from pydub.playback import play
from classification.tools import *

## Given an audio file and the start/end times for a detection event, create a new audio file that extracts the detection event
# file_in - path to the original file that contains the detection event audio data
# path_out - path to the output file that will be created
# date_detection_start, date_detection_end - datetime objects
# buffer - seconds of audio to include before and after the detection
def extract_detection_audio(file_in, dir_out, date_detection_start, date_detection_end, tag='', buffer=1.0):

    info = get_info_from_filename(file_in)
    date_file = datetime.datetime(int(info['year']), int(info['month']), int(info['day']), int(info['hour']), int(info['min']), int(info['sec']))

    file_out = f"{tag}_{info['serial_no']}_{date_detection_start.strftime('%Y%m%d_%H%M%S')}.wav"
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

    # Save the extracted file
    # play(detection_segment)
    path_out = f'{dir_out}/{file_out}'
    print(f'Saving to {path_out}')
    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))
    detection_segment.export(path_out, format='wav')
