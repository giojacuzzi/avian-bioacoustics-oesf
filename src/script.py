print('start')

import os
import subprocess
import tempfile
import time

def launch_terminal_process(working_dir, python_path, script_path, arguments, callback):

    done_signal_file = tempfile.NamedTemporaryFile(delete=False)
    done_signal_path = done_signal_file.name
    done_signal_file.close()
    command = (
        f'osascript -e \'tell application "Terminal" to activate\' '
        f'-e \'tell application "Terminal" to do script '
        f'"cd \\"{os.path.abspath(working_dir)}\\" && '
        f'ls && {os.path.abspath(python_path)} \\"{script_path}\\" {arguments}'
        f'; rm \\"{done_signal_path}\\""\''
    )
    subprocess.run(command, shell=True)
    while os.path.exists(done_signal_path):
        time.sleep(1)
    callback()

def on_process_finish():
    print('on_process_finish callback')

in_path = '/Users/giojacuzzi/Desktop/audio_test_files/chorus/chorus1.wav'
in_filetype = '.wav'
out_dir_path = '/Users/giojacuzzi/Downloads/output'
out_filetype = '.csv'
retain_dir_tree = True
source_labels_filepath = 'data/species_list_OESF.txt'
target_model_filepath = 'data/models/custom/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0.tflite'
target_labels_filepath = 'data/models/custom/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0_Labels.txt'
use_ensemble = True
ensemble_class_model_selections = 'data/ensemble/class_model_selections.csv'
min_confidence = 0.25
retain_logit_score = False
n_processes = 8
n_separation = 1
digits = 3

launch_terminal_process(
    working_dir="",
    python_path=".venv/bin/python",
    script_path="src/run_process_audio_script.py",
    arguments=" ".join([
        in_path,
        in_filetype,
        out_dir_path,
        out_filetype,
        "--retain_dir_tree" if retain_dir_tree else "",
        "--source_labels_filepath", source_labels_filepath,
        "--target_model_filepath", target_model_filepath,
        "--target_labels_filepath", target_labels_filepath,
        "--use_ensemble" if use_ensemble else "",
        "--ensemble_class_model_selections", ensemble_class_model_selections,
        "--min_confidence", str(min_confidence),
        "--retain_logit_score" if retain_logit_score else "",
        "--n_processes", str(n_processes),
        "--n_separation", str(n_separation),
        "--digits", str(digits)
    ]),
    callback=on_process_finish
)

print('end')