import tkinter
import customtkinter
from tkinterdnd2 import *

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

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        # Variables
        self.retain_dir_tree = tkinter.IntVar(self, 1)

        self.title("Acoustic classifier model interface")
        self.geometry("800x500")
        # self.grid_columnconfigure((0,1), weight=1)
        # self.grid_rowconfigure((0,1,2), weight=1)

        # Frames
        self.frame_io = customtkinter.CTkFrame(self, fg_color='red')
        self.frame_config = customtkinter.CTkFrame(self, fg_color='blue')
        # self.frame_process = customtkinter.CTkFrame(self, fg_color='black')

        self.frame_io.pack(side='left', fill = 'both', expand = True, padx=5, pady=10)
        self.frame_config.pack(side='left', fill = 'both', expand = True, padx=5, pady=10)
        # self.frame_process.pack(side='left', fill = 'both', expand = True, padx=5, pady=10)

        self.frame_input = customtkinter.CTkFrame(self.frame_io, fg_color='purple')
        self.frame_output = customtkinter.CTkFrame(self.frame_io, fg_color='yellow')
        self.frame_input.pack(side='top', fill = 'both', expand = True, padx=10, pady=5)
        self.frame_output.pack(side='top', fill = 'both', expand = True, padx=10, pady=5)

        self.frame_models = customtkinter.CTkFrame(self.frame_config, fg_color='orange')
        self.frame_options = customtkinter.CTkFrame(self.frame_config, fg_color='green')
        self.frame_models.pack(side='top', fill = 'both', expand = True, padx=10, pady=5)
        self.frame_options.pack(side='top', fill = 'both', expand = True, padx=10, pady=5)

        # Input
        self.label_input = customtkinter.CTkLabel(self.frame_input, text="Input audio data")
        self.label_input.pack(side='top')
        self.entry_in_path = customtkinter.CTkEntry(self.frame_input, placeholder_text="Type, open, or drag and drop an audio file or directory...")
        self.entry_in_path.drop_target_register(DND_FILES)
        self.entry_in_path.dnd_bind('<<Drop>>', self.callback_entry_in_path_dnd)
        self.entry_in_path.pack(side='top', fill = 'x', padx=10, pady=5)
        self.button_open_in_file = customtkinter.CTkButton(self.frame_input, text='Open file', command=self.callback_button_open_in_path_file)
        self.button_open_in_file.pack(side='top', padx=10, pady=5)
        self.button_open_in_dir = customtkinter.CTkButton(self.frame_input, text='Open directory', command=self.callback_button_open_in_path_dir)
        self.button_open_in_dir.pack(side='top', padx=10, pady=5)
        self.option_in_filetype = customtkinter.CTkOptionMenu(self.frame_input, dynamic_resizing=True, values=['.wav', '.aif', '.flac', '.mp3'])
        self.option_in_filetype.pack(side='top', padx=10, pady=5)
    
        # Output
        self.label_output = customtkinter.CTkLabel(self.frame_output, text="Output model predictions")
        self.label_output.pack(side='top')
        self.entry_out_dir_path = customtkinter.CTkEntry(self.frame_output, placeholder_text="Type, open, or drag and drop a directory...")
        self.entry_out_dir_path.drop_target_register(DND_FILES)
        self.entry_out_dir_path.dnd_bind('<<Drop>>', self.callback_entry_out_dir_path_dnd)
        self.entry_out_dir_path.pack(side='top', fill = 'x', padx=10, pady=5)
        self.button_open_out_dir = customtkinter.CTkButton(self.frame_output, text='Open directory', command=self.callback_button_open_out_path_dir)
        self.button_open_out_dir.pack(side='top', padx=10, pady=5)
        self.option_out_filetype = customtkinter.CTkOptionMenu(self.frame_output, dynamic_resizing=True, values=['.csv', '.parquet'])
        self.option_out_filetype.pack(side='top', padx=10, pady=5)
        self.checkbox_retain_dir_tree = customtkinter.CTkCheckBox(self.frame_output, variable=self.retain_dir_tree, onvalue=1, offvalue=0, command=self.callback_checkbox_retain_dir_tree, text='Retain directory tree')
        self.checkbox_retain_dir_tree.pack(side='top', padx=10, pady=5)

        # Models
        self.label_model = customtkinter.CTkLabel(self.frame_models, text="Model configuration")
        self.label_model.pack(side='top')
    
        # Options
        self.label_options = customtkinter.CTkLabel(self.frame_options, text="Options")
        self.label_options.pack(side='top')

    # Input callbacks
    def callback_entry_in_path_dnd(self, event):
        print(event.data)
        self.entry_in_path.delete(0,tkinter.END)
        self.entry_in_path.insert('0', event.data)

    def callback_button_open_in_path_file(self):
        path = customtkinter.filedialog.askopenfilename()
        print(path)
        self.entry_in_path.delete(0,tkinter.END)
        self.entry_in_path.insert('0', path)

    def callback_button_open_in_path_dir(self):
        path = customtkinter.filedialog.askdirectory()
        print(path)
        self.entry_in_path.delete(0,tkinter.END)
        self.entry_in_path.insert('0', path)

    # Output callbacks
    def callback_entry_out_dir_path_dnd(self, event):
        print(event.data)
        self.entry_out_dir_path.delete(0,tkinter.END)
        self.entry_out_dir_path.insert('0', event.data)

    def callback_button_open_out_path_dir(self):
        path = customtkinter.filedialog.askdirectory()
        print(path)
        self.entry_out_dir_path.delete(0,tkinter.END)
        self.entry_out_dir_path.insert('0', path)
    
    def callback_checkbox_retain_dir_tree(self):
        print("callback_checkbox_retain_dir_tree:", self.retain_dir_tree.get())

    # Other callbacks
    def button_callback(self):
        print('button_callback START')
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
        print('button_callback END')

app = App()
app.mainloop()