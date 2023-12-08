import os
import subprocess
import helpers
import sys
import shutil
from pydub import AudioSegment

# Temporary folder for separation files
# Returns a list of all file paths, including a copy of the original
path_temp = 'annotation/_output/temp/'

def get_output_path():
    return(os.path.abspath(path_temp))

def separate(path, num_sources = 4, multichannel = False):

    # Remove any existing temp files
    if os.path.exists(path_temp):
        shutil.rmtree(path_temp)
    os.makedirs(path_temp)

    path_file_stub = os.path.splitext(os.path.basename(path))[0]

    path_local_copy = f'{path_temp}{os.path.basename(path)}'
    w = AudioSegment.from_wav(path)
    w = helpers.remove_dc_offset(w)
    w.export(path_local_copy, format='wav')

    if num_sources == 4:
        model_dir = '../sound-separation/models/bird_mixit/bird_mixit_model_checkpoints/output_sources4'
        checkpoint = '../sound-separation/models/bird_mixit/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090'
    if num_sources == 8:
        model_dir = '../sound-separation/models/bird_mixit/bird_mixit_model_checkpoints/output_sources8'
        checkpoint = '../sound-separation/models/bird_mixit/bird_mixit_model_checkpoints/output_sources8/model.ckpt-2178900'

    print(f'Separating {os.path.basename(path_local_copy)} into {num_sources} sources...')
    subprocess.run([
        'python', '../sound-separation/models/tools/process_wav.py',
        '--model_dir', model_dir,
        '--checkpoint', checkpoint,
        '--num_sources', str(num_sources),
        '--input', path_local_copy,
        '--output', f'{path_temp}{path_file_stub}.wav'
    ])

    # Get absolute paths to all files
    files = sorted(os.listdir(path_temp))
    for i in range(len(files)):
        files[i] = os.path.dirname(os.path.abspath(files[i])) + '/' + path_temp + os.path.basename(files[i])

    if multichannel:
        path_multichannel = f'{path_temp}{path_file_stub}_{1+num_sources}ch.wav'
        channels = []
        for f in files:
            print(f)
            channels.append(AudioSegment.from_wav(f))
        if num_sources == 4:
            multichannel_file = AudioSegment.from_mono_audiosegments(channels[0], channels[1], channels[2], channels[3], channels[4])
        elif num_sources == 8:
             multichannel_file = AudioSegment.from_mono_audiosegments(channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8])
        multichannel_file.export(path_multichannel, format='wav')
        # shutil.rmtree(path_temp) # cleanup individual source files
        for f in files:
            os.remove(f)
        files = [os.path.dirname(os.path.abspath(path_multichannel)) + '/' + path_temp + os.path.basename(path_multichannel)]

    return(files)
