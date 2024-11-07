if __name__ == '__main__':

    from audio import process_audio

    # in_filepath = input('\033[34mDrag and drop file to analyze (requires full path): \033[0m')
    # # Normalize file paths to support both mac and windows
    # if in_filepath.startswith("'") and in_filepath.endswith("'"):
    #     in_filepath = in_filepath[1:-1]
    # in_filepath = os.path.normpath(in_filepath)

    ## Input
    in_path = '/Users/giojacuzzi/Desktop/audio_test_files/chorus' # Full path to a single audio file or a directory containing audio files. Will search the directory tree recursively.
    in_filetype = '.wav' # Supported file types: '.wav', '.aif', '.mp3'

    ## Output
    out_dir_path = '/Users/giojacuzzi/Downloads/output' # Full path to the output directory
    out_filetype = '.csv' # .csv (human-readable, larger storage size) or .parquet (compressed, smaller storage size)
    retain_dir_tree = True # Retain the original directory tree structure relative to input directory. If set to False, will output all files to the same directory.
    sort_by = ['start_time', 'confidence'] # Column(s) to sort dataframe by, e.g. 'confidence' or ['start_time','common_name'] or '' for no sorting
    ascending = [True, False] # Sort direction(s)
    digits = 3

    ## Model configuration
    # Source model
    source_labels_filepath = 'src/classification/species_list/species_list_OESF.txt'
    # Target model
    use_custom_model = True
    if use_custom_model:
        custom_model_filepath = 'data/models/custom/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0.tflite'
        target_labels_filepath = 'data/models/custom/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0_Labels.txt'
    else:
        custom_model_filepath = None
        target_labels_filepath = None
    # Ensemble
    use_ensemble = True
    if use_ensemble:
        ensemble_class_model_selections = 'data/ensemble/class_model_selections.csv'
    else:
        ensemble_class_model_selections = None

    ## Analysis configuration
    n_processes = 8 # Number of cores used by the processing pool (<= number of physical cores available on your computer)
    num_separation = 1 # Number of channels to separate for analysis. Leave as 1 for original file alone.
    min_confidence = 0.25 # Minimum confidence score to retain a detection # TODO
    retain_logit_score = False

    process_audio.process_file_or_dir(
        in_path                         = in_path,
        in_filetype                     = in_filetype,
        out_dir_path                    = out_dir_path,
        retain_dir_tree                 = retain_dir_tree,
        custom_model_filepath           = custom_model_filepath,
        source_labels_filepath          = source_labels_filepath,
        target_labels_filepath          = target_labels_filepath,
        use_ensemble                    = use_ensemble,
        ensemble_class_model_selections = ensemble_class_model_selections,
        min_confidence                  = min_confidence,
        retain_logit_score              = retain_logit_score,
        n_processes                     = n_processes,
        num_separation                  = num_separation,
        cleanup                         = True,
        sort_by                         = sort_by,
        ascending                       = ascending,
        out_filetype                    = out_filetype,
        digits                          = digits
    )
