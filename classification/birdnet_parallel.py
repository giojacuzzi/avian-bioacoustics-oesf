import pandas as pd
import glob
import os
from birdnetlib.batch import DirectoryMultiProcessingAnalyzer 
from birdnetlib.analyzer import Analyzer
import datetime

def on_analyze_directory_complete(recordings):
        print(f'Directory complete ({len(recordings)} recordings analyzed)')
        for recording in recordings:
            # print(recording.detections)
            print('This is a recording!')

def getDirectoriesWithFiles(path, filetype):
    directoryList = []
    if os.path.isfile(path):
        return []
    # Add dir to directorylist if it contains files of filetype
    if len([f for f in os.listdir(path) if f.endswith(filetype)]) > 0:
        directoryList.append(path)
    for d in os.listdir(path):
        new_path = os.path.join(path, d)
        if os.path.isdir(new_path):
            directoryList += getDirectoriesWithFiles(new_path, filetype)
    return directoryList

if __name__ == '__main__':
    analyzer = Analyzer()

    # File config
    in_filetype = '.wav'
    out_dir = os.path.dirname(__file__) + '\\_output\\'
    out_file = 'detections.csv'
    out_path = out_dir + out_file
    # in_dir = "D:\\DNR\\2021\\Deployment1_2021April13_14\\S4A04325_20210414_Data"
    in_dir = 'C:\\Users\\gioj\\Desktop\\test'

    # for dir in getDirectoriesWithFiles(in_dir, in_filetype):
    #     if __name__ == '__main__':
    #         print('Analyzing directory ' + dir + '...')

    #         batch = DirectoryMultiProcessingAnalyzer(
    #             directory=dir,
    #             analyzers = [analyzer],
    #             patterns=['*' + in_filetype]
    #         )
    #         batch.on_analyze_directory_complete = on_analyze_directory_complete

    #         print('Processing...')
    #         batch.process()
    #         print('Done!')

    # print('FINITO!')

    dir = getDirectoriesWithFiles(in_dir, in_filetype)[0]
    batch = DirectoryMultiProcessingAnalyzer(
        directory=dir,
        analyzers = [analyzer],
        patterns=["*.wav"]
    )
    batch.on_analyze_directory_complete = on_analyze_directory_complete
    print('Processing...')
    batch.process()
    print('Done!')

# ##########

# # Load csv of processed files

# # For each directory containing .wav files
#     # If the files in the directory haven't been processed
#         # Create a batch for the directory
#         # Process the batch
#         # Save the results