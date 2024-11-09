import tkinter
import customtkinter
from tkinterdnd2 import *
from CTkMenuBar import *

import os
import subprocess
import tempfile
import time
import json

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

        menu = CTkMenuBar(master=self, bg_color='#222222')
        menu_button_file = menu.add_cascade("File")
        menu_button_about = menu.add_cascade("About", postcommand=self.callback_about_popup)
        menu_dropdown_file = CustomDropdownMenu(widget=menu_button_file)
        menu_dropdown_file.add_option(option="Open session...", command=self.open_session_file)
        menu_dropdown_file.add_option(option="Save session", command=self.save_session_file)

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
        self.button_open_in_dir = customtkinter.CTkButton(self.frame_input_config, text='Open directory', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.callback_button_open_in_path_dir)
        self.button_open_in_dir.pack(side='left', padx=10, pady=5)
        self.button_open_in_file = customtkinter.CTkButton(self.frame_input_config, text='Open file', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.callback_button_open_in_path_file)
        self.button_open_in_file.pack(side='left', padx=10, pady=5)
    
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
        self.entry_source_labels_filepath.drop_target_register(DND_FILES)
        self.entry_source_labels_filepath.dnd_bind('<<Drop>>', self.callback_entry_source_labels_filepath_dnd)
        self.switch_use_target_model = customtkinter.CTkSwitch(self.tabview_model.tab('Target model'), variable=self.use_target_model, onvalue=1, offvalue=0, command=self.callback_switch_use_target_model, text='Use target model')
        self.switch_use_target_model.select()
        self.switch_use_target_model.pack(side='top', padx=10, pady=5)
        self.label_model = customtkinter.CTkLabel(self.tabview_model.tab('Target model'), text="Model file")
        self.label_model.pack(side='top')
        self.entry_target_model_filepath = customtkinter.CTkEntry(self.tabview_model.tab('Target model'), placeholder_text="Path to target model .tflite file")
        self.entry_target_model_filepath.pack(side='top', fill = 'x', padx=10, pady=5)
        self.entry_target_model_filepath.drop_target_register(DND_FILES)
        self.entry_target_model_filepath.dnd_bind('<<Drop>>', self.callback_entry_target_model_filepath_dnd)
        self.label_model = customtkinter.CTkLabel(self.tabview_model.tab('Target model'), text="Class labels")
        self.label_model.pack(side='top')
        self.entry_target_labels_filepath = customtkinter.CTkEntry(self.tabview_model.tab('Target model'), placeholder_text="Path to target model labels .txt file")
        self.entry_target_labels_filepath.pack(side='top', fill = 'x', padx=10, pady=5)
        self.entry_target_labels_filepath.drop_target_register(DND_FILES)
        self.entry_target_labels_filepath.dnd_bind('<<Drop>>', self.callback_entry_target_labels_filepath_dnd)
        self.switch_use_ensemble = customtkinter.CTkSwitch(self.tabview_model.tab('Ensemble'), variable=self.use_ensemble, onvalue=1, offvalue=0, command=self.callback_switch_use_ensemble, text='Use ensemble')
        self.switch_use_ensemble.select()
        self.switch_use_ensemble.pack(side='top', padx=10, pady=5)
        self.label_model = customtkinter.CTkLabel(self.tabview_model.tab('Ensemble'), text="Class model selections")
        self.label_model.pack(side='top')
        self.entry_ensemble_class_model_selections = customtkinter.CTkEntry(self.tabview_model.tab('Ensemble'), placeholder_text="Path to class model selections .csv file")
        self.entry_ensemble_class_model_selections.pack(side='top', fill = 'x', padx=10, pady=5)
        self.entry_ensemble_class_model_selections.drop_target_register(DND_FILES)
        self.entry_ensemble_class_model_selections.dnd_bind('<<Drop>>', self.callback_entry_ensemble_class_model_selections_dnd)
    
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
        self.frame_round = customtkinter.CTkFrame(self.frame_options, fg_color='transparent')
        self.frame_round.pack(side='top', fill='x')
        self.label_options = customtkinter.CTkLabel(self.frame_round, text="Rounding")
        self.label_options.pack(side='left', padx=10)
        self.spinbox_round = spinbox.Spinbox(self.frame_round, type='int', min=1, max=10, width=150, step_size=1)
        self.spinbox_round.pack(side='right', padx=10, pady=5)
        self.spinbox_round.set(3)
        self.checkbox_retain_logit_score = customtkinter.CTkCheckBox(self.frame_options, variable=self.retain_logit_score, onvalue=1, offvalue=0, command=self.callback_checkbox_retain_logit_score, text='Retain logit score')
        self.checkbox_retain_logit_score.deselect()
        self.checkbox_retain_logit_score.pack(side='top', padx=10, pady=5)

        # Prime necessary callbacks
        self.callback_switch_use_target_model()
    
    # Menu callbacks
    def open_session_file(self):
        print('open_session_file')
        path = customtkinter.filedialog.askopenfilename()
        if path != '':
            with open(path, 'r') as f:
                session = json.load(f)
                self.entry_in_path.delete(0,tkinter.END)
                self.entry_in_path.insert('0', session['in_path'])
                self.option_in_filetype.set(session['in_filetype'])
                self.entry_out_dir_path.delete(0,tkinter.END)
                self.entry_out_dir_path.insert('0', session['out_dir_path'])
                self.option_out_filetype.set(session['out_filetype'])
                if session['retain_dir_tree']:
                    self.checkbox_retain_dir_tree.select()
                else:
                    self.checkbox_retain_dir_tree.deselect()
                self.entry_source_labels_filepath.delete(0,tkinter.END)
                self.entry_source_labels_filepath.insert('0', session['source_labels_filepath'])
                if session['use_target_model']:
                    self.switch_use_target_model.select()
                else:
                    self.switch_use_target_model.deselect()
                self.entry_target_model_filepath.delete(0,tkinter.END)
                self.entry_target_model_filepath.insert('0', session['target_model_filepath'])
                self.entry_target_labels_filepath.delete(0,tkinter.END)
                self.entry_target_labels_filepath.insert('0', session['target_labels_filepath'])
                if session['use_ensemble']:
                    self.switch_use_ensemble.select()
                else:
                    self.switch_use_ensemble.deselect()
                self.entry_ensemble_class_model_selections.delete(0,tkinter.END)
                self.entry_ensemble_class_model_selections.insert('0', session['ensemble_class_model_selections'])
                self.spinbox_n_processes.set(session['n_processes'])
                self.spinbox_n_separation.set(session['n_separation'])
                self.spinbox_min_confidence.set(session['min_confidence'])
                self.spinbox_round.set(session['round'])
                if session['retain_logit_score']:
                    self.checkbox_retain_logit_score.select()
                else:
                    self.checkbox_retain_logit_score.deselect()
    
    def save_session_file(self):
        print('save_session_file')
        path = customtkinter.filedialog.askdirectory()
        dialog = customtkinter.CTkInputDialog(text="Filename: (e.g. session.json)", title="Save session")
        filepath = f'{path}/{dialog.get_input()}'
        session = {
            'in_path' : self.entry_in_path.get(),
            'in_filetype' : self.option_in_filetype.get(),
            'out_dir_path' : self.entry_out_dir_path.get(),
            'out_filetype' : self.option_out_filetype.get(),
            'retain_dir_tree' : self.retain_dir_tree.get(),
            'source_labels_filepath' : self.entry_source_labels_filepath.get(),
            'use_target_model' : self.switch_use_target_model.get(),
            'target_model_filepath' : self.entry_target_model_filepath.get(),
            'target_labels_filepath' : self.entry_target_labels_filepath.get(),
            'use_ensemble' : self.use_ensemble.get(),
            'ensemble_class_model_selections' : self.entry_ensemble_class_model_selections.get(),
            'n_processes' : self.spinbox_n_processes.get(),
            'n_separation' : self.spinbox_n_separation.get(),
            'min_confidence' : self.spinbox_min_confidence.get(),
            'round' : self.spinbox_round.get(),
            'retain_logit_score' : self.retain_logit_score.get()
        }
        with open(filepath, 'w') as f:
            json.dump(session, f, indent=4)
    
    def callback_about_popup(self):
        global about_popup
        about_popup = customtkinter.CTkToplevel(self)
        about_popup.title('About')
        about_popup.geometry('1100x170')
        about_popup_textbox = customtkinter.CTkTextbox(about_popup)
        about_popup_textbox.pack(side='left', fill='both', expand=True, padx=0, pady=0)
        about_popup_textbox.insert("0.0", """
        This software is provided free and open-source under a BSD-3-Clause license. If you use it for your research, please cite as:\n\n
            Jacuzzi, G., Olden, J.D. et al. Few-shot transfer learning enables robust acoustic monitoring of wildlife communities at the landscape scale. (in preparation).\n\n
        Copyright (c) 2024, Giordano Jacuzzi.
        """)
        about_popup_textbox.configure(state="disabled")

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
    def callback_switch_use_target_model(self):
        value = self.use_target_model.get()
        print("callback_checkbox_use_target_model:", value)
    
    def callback_entry_source_labels_filepath_dnd(self, event):
        print(event.data)
        self.entry_source_labels_filepath.delete(0,tkinter.END)
        self.entry_source_labels_filepath.insert('0', event.data)
    
    def callback_entry_target_model_filepath_dnd(self, event):
        print(event.data)
        self.entry_target_model_filepath.delete(0,tkinter.END)
        self.entry_target_model_filepath.insert('0', event.data)
    
    def callback_entry_target_labels_filepath_dnd(self, event):
        print(event.data)
        self.entry_target_labels_filepath.delete(0,tkinter.END)
        self.entry_target_labels_filepath.insert('0', event.data)

    def callback_entry_ensemble_class_model_selections_dnd(self, event):
        print(event.data)
        self.entry_ensemble_class_model_selections.delete(0,tkinter.END)
        self.entry_ensemble_class_model_selections.insert('0', event.data)

    def callback_switch_use_ensemble(self):
        value = self.use_ensemble.get()
        print("callback_checkbox_use_ensemble:", value)
    
    # Options callbacks
    def callback_checkbox_retain_logit_score(self):
        print("callback_checkbox_retain_logit_score:", self.retain_logit_score.get())

    # Other callbacks
    def callback_button_launch_process(self):
        print('button_callback START')
        in_path = self.entry_in_path.get()
        in_filetype = self.option_in_filetype.get()
        out_dir_path = self.entry_out_dir_path.get()
        out_filetype = self.option_out_filetype.get()
        retain_dir_tree = self.retain_dir_tree.get()
        source_labels_filepath = self.entry_source_labels_filepath.get()
        target_model_filepath = self.entry_target_model_filepath.get()
        target_labels_filepath = self.entry_target_labels_filepath.get()
        use_ensemble = self.use_ensemble.get()
        ensemble_class_model_selections = self.entry_ensemble_class_model_selections.get()
        min_confidence = self.spinbox_min_confidence.get()
        retain_logit_score = self.retain_logit_score.get()
        n_processes = self.spinbox_n_processes.get()
        n_separation = self.spinbox_n_separation.get()
        round = self.spinbox_round.get()
        # in_path = '/Users/giojacuzzi/Desktop/audio_test_files/chorus/chorus1.wav'
        # in_filetype = '.wav'
        # out_dir_path = '/Users/giojacuzzi/Downloads/output'
        # out_filetype = '.csv'
        # retain_dir_tree = True
        # source_labels_filepath = 'data/species_list_OESF.txt'
        # target_model_filepath = 'data/models/custom/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0.tflite'
        # target_labels_filepath = 'data/models/custom/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0_Labels.txt'
        # use_ensemble = True
        # ensemble_class_model_selections = 'data/ensemble/class_model_selections.csv'
        # min_confidence = 0.25
        # retain_logit_score = False
        # n_processes = 8
        # n_separation = 1
        # round = 3

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
                "--digits", str(round)
            ]),
            callback=on_process_finish
        )
        print('button_callback END')

app = App()
app.mainloop()