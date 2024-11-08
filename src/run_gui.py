import tkinter
import customtkinter
from tkinterdnd2 import *

import os
import subprocess
import tempfile
import time

import gui.gui_spinbox as spinbox

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

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        # Variables
        self.retain_dir_tree = tkinter.IntVar(self, 1)
        self.retain_logit_score = tkinter.IntVar(self, 1)
        self.use_target_model = tkinter.IntVar(self, 1)
        self.use_ensemble = tkinter.IntVar(self, 1)

        self.title("Acoustic classifier model interface")
        self.geometry("1180x300")

        # Frames
        self.frame_io = customtkinter.CTkFrame(self, fg_color='transparent')
        self.frame_config = customtkinter.CTkFrame(self)
        # self.frame_process = customtkinter.CTkFrame(self, fg_color='black')

        self.frame_io.pack(side='left', fill = 'both', expand = True, padx=5, pady=5)
        self.frame_config.pack(side='left', fill = 'both', expand = False, padx=5, pady=5)
        # self.tabview_model.pack(side='top')
        # self.frame_process.pack(side='left', fill = 'both', expand = True, padx=5, pady=10)

        self.frame_input = customtkinter.CTkFrame(self.frame_io)
        self.frame_output = customtkinter.CTkFrame(self.frame_io)
        self.frame_input.pack(side='top', fill = 'both', expand = True, padx=0, pady=(0,5))
        self.frame_output.pack(side='top', fill = 'both', expand = True, padx=0, pady=(5,0))

        # self.frame_models = customtkinter.CTkFrame(self.frame_config, fg_color='orange')
        self.tabview_model = customtkinter.CTkTabview(self.frame_config)
        self.frame_options = customtkinter.CTkFrame(self.frame_config)
        self.tabview_model.pack(side='left', fill = 'both', expand = True, padx=10, pady=5)
        self.frame_options.pack(side='left', fill = 'both', expand = True, padx=10, pady=5)

        # Input
        self.label_input = customtkinter.CTkLabel(self.frame_input, text="Input audio data")
        self.label_input.pack(side='top')
        self.entry_in_path = customtkinter.CTkEntry(self.frame_input, placeholder_text="Path to audio file or directory")
        self.entry_in_path.drop_target_register(DND_FILES)
        self.entry_in_path.dnd_bind('<<Drop>>', self.callback_entry_in_path_dnd)
        self.entry_in_path.pack(side='top', fill = 'x', padx=10, pady=5)
        self.frame_input_config = customtkinter.CTkFrame(self.frame_input, fg_color='transparent')
        self.frame_input_config.pack(side='top')
        self.option_in_filetype = customtkinter.CTkOptionMenu(self.frame_input_config, dynamic_resizing=True, values=['.wav', '.aif', '.flac', '.mp3'])
        self.option_in_filetype.pack(side='left', padx=10, pady=5)
        self.button_open_in_file = customtkinter.CTkButton(self.frame_input_config, text='Open file', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.callback_button_open_in_path_file)
        self.button_open_in_file.pack(side='left', padx=10, pady=5)
        self.button_open_in_dir = customtkinter.CTkButton(self.frame_input_config, text='Open directory', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.callback_button_open_in_path_dir)
        self.button_open_in_dir.pack(side='left', padx=10, pady=5)
    
        # Output
        self.label_output = customtkinter.CTkLabel(self.frame_output, text="Output model predictions")
        self.label_output.pack(side='top')
        self.entry_out_dir_path = customtkinter.CTkEntry(self.frame_output, placeholder_text="Path to directory")
        self.entry_out_dir_path.drop_target_register(DND_FILES)
        self.entry_out_dir_path.dnd_bind('<<Drop>>', self.callback_entry_out_dir_path_dnd)
        self.entry_out_dir_path.pack(side='top', fill = 'x', padx=10, pady=5)
        self.frame_output_config = customtkinter.CTkFrame(self.frame_output, fg_color='transparent')
        self.frame_output_config.pack(side='top')
        self.option_out_filetype = customtkinter.CTkOptionMenu(self.frame_output_config, dynamic_resizing=True, values=['.csv', '.parquet'])
        self.option_out_filetype.pack(side='left', padx=10, pady=5)
        self.button_open_out_dir = customtkinter.CTkButton(self.frame_output_config, text='Open directory', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.callback_button_open_out_path_dir)
        self.button_open_out_dir.pack(side='left', padx=10, pady=5)
        self.checkbox_retain_dir_tree = customtkinter.CTkCheckBox(self.frame_output_config, variable=self.retain_dir_tree, onvalue=1, offvalue=0, command=self.callback_checkbox_retain_dir_tree, text='Retain directory tree')
        self.checkbox_retain_dir_tree.select()
        self.checkbox_retain_dir_tree.pack(side='left', padx=10, pady=5)
        self.button_launch_process = customtkinter.CTkButton(self.frame_output, text='Launch process', command=self.callback_button_launch_process)
        self.button_launch_process.pack(side='bottom', fill = 'x', padx=10, pady=5)

        # Models
        # self.label_model = customtkinter.CTkLabel(self.frame_models, text="Model configuration")
        # self.label_model.pack(side='top')
        # self.tabview_model = customtkinter.CTkTabview(self.frame_models)
        # self.tabview_model.pack(side='top')
        self.tabview_model.add('Source model')
        self.tabview_model.add('Target model')
        self.tabview_model.add('Ensemble')
        self.label_model = customtkinter.CTkLabel(self.tabview_model.tab('Source model'), text="Class labels")
        self.label_model.pack(side='top')
        self.entry_source_labels_filepath = customtkinter.CTkEntry(self.tabview_model.tab('Source model'), placeholder_text="Path to source model labels .txt file")
        self.entry_source_labels_filepath.pack(side='top', fill = 'x', padx=10, pady=5)
        self.checkbox_use_target_model = customtkinter.CTkCheckBox(self.tabview_model.tab('Target model'), variable=self.use_target_model, onvalue=1, offvalue=0, command=self.callback_checkbox_use_target_model, text='Use target model')
        self.checkbox_use_target_model.select()
        self.checkbox_use_target_model.pack(side='top', padx=10, pady=5)
        self.label_model = customtkinter.CTkLabel(self.tabview_model.tab('Target model'), text="Model file")
        self.label_model.pack(side='top')
        self.entry_target_model_filepath = customtkinter.CTkEntry(self.tabview_model.tab('Target model'), placeholder_text="Path to target model .tflite file")
        self.entry_target_model_filepath.pack(side='top', fill = 'x', padx=10, pady=5)
        self.label_model = customtkinter.CTkLabel(self.tabview_model.tab('Target model'), text="Class labels")
        self.label_model.pack(side='top')
        self.entry_target_labels_filepath = customtkinter.CTkEntry(self.tabview_model.tab('Target model'), placeholder_text="Path to target model labels .txt file")
        self.entry_target_labels_filepath.pack(side='top', fill = 'x', padx=10, pady=5)
        self.checkbox_use_ensemble = customtkinter.CTkCheckBox(self.tabview_model.tab('Ensemble'), variable=self.use_ensemble, onvalue=1, offvalue=0, command=self.callback_checkbox_use_ensemble, text='Use ensemble')
        self.checkbox_use_ensemble.select()
        self.checkbox_use_ensemble.pack(side='top', padx=10, pady=5)
        self.label_model = customtkinter.CTkLabel(self.tabview_model.tab('Ensemble'), text="Class model selections")
        self.label_model.pack(side='top')
        self.entry_ensemble_class_model_selections = customtkinter.CTkEntry(self.tabview_model.tab('Ensemble'), placeholder_text="Path to class model selections .csv file")
        self.entry_ensemble_class_model_selections.pack(side='top', fill = 'x', padx=10, pady=5)
    
        # Options
        self.label_options = customtkinter.CTkLabel(self.frame_options, text="Processing options")
        self.label_options.pack(side='top')
        self.frame_n_processes = customtkinter.CTkFrame(self.frame_options, fg_color='transparent')
        self.frame_n_processes.pack(side='top', fill='x')
        self.label_options = customtkinter.CTkLabel(self.frame_n_processes, text="Processes")
        self.label_options.pack(side='left', padx=10)
        self.spinbox_n_processes = spinbox.Spinbox(self.frame_n_processes, type='int', min=1, max=128, width=150, step_size=1)
        self.spinbox_n_processes.pack(side='right', padx=10, pady=5)
        self.spinbox_n_processes.set(8)
        self.frame_n_separation = customtkinter.CTkFrame(self.frame_options, fg_color='transparent')
        self.frame_n_separation.pack(side='top', fill='x')
        self.label_options = customtkinter.CTkLabel(self.frame_n_separation, text="Separation")
        self.label_options.pack(side='left', padx=10)
        self.spinbox_n_separation = spinbox.Spinbox(self.frame_n_separation, type='int', min=1, max=8, width=150, step_size=1)
        self.spinbox_n_separation.pack(side='right', padx=10, pady=5)
        self.spinbox_n_separation.set(1)
        self.frame_min_confidence = customtkinter.CTkFrame(self.frame_options, fg_color='transparent')
        self.frame_min_confidence.pack(side='top', fill='x')
        self.label_options = customtkinter.CTkLabel(self.frame_min_confidence, text="Min confidence")
        self.label_options.pack(side='left', padx=10)
        self.spinbox_min_confidence = spinbox.Spinbox(self.frame_min_confidence, type='float', min=0.0, max=1.0, width=150, step_size=0.01)
        self.spinbox_min_confidence.pack(side='right', padx=10, pady=5)
        self.spinbox_min_confidence.set(0.1)
        self.frame_digits = customtkinter.CTkFrame(self.frame_options, fg_color='transparent')
        self.frame_digits.pack(side='top', fill='x')
        self.label_options = customtkinter.CTkLabel(self.frame_digits, text="Rounding")
        self.label_options.pack(side='left', padx=10)
        self.spinbox_digits = spinbox.Spinbox(self.frame_digits, type='int', min=1, max=10, width=150, step_size=1)
        self.spinbox_digits.pack(side='right', padx=10, pady=5)
        self.spinbox_digits.set(3)
        self.checkbox_retain_logit_score = customtkinter.CTkCheckBox(self.frame_options, variable=self.retain_logit_score, onvalue=1, offvalue=0, command=self.callback_checkbox_retain_logit_score, text='Retain logit score')
        self.checkbox_retain_logit_score.deselect()
        self.checkbox_retain_logit_score.pack(side='top', padx=10, pady=5)

        # Prime necessary callbacks
        self.callback_checkbox_use_target_model()

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

    # Model config callbacks
    def callback_checkbox_use_target_model(self):
        value = self.use_target_model.get()
        print("callback_checkbox_use_target_model:", value)
        # if value == 1:
        #     self.entry_target_model_filepath.configure(state= "normal")
        # else:
        #     self.entry_target_model_filepath.configure(state= "disabled")

    def callback_checkbox_use_ensemble(self):
        value = self.use_ensemble.get()
        print("callback_checkbox_use_ensemble:", value)
    
    # Options callbacks
    def callback_checkbox_retain_logit_score(self):
        print("callback_checkbox_retain_logit_score:", self.retain_logit_score.get())

    # Other callbacks
    def callback_button_launch_process(self):
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