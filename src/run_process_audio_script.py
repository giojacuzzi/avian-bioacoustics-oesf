import argparse
from audio import process_audio

def main(args):
    print('run_process_audio_script START')

    # Required arguments
    print(f"in_path: {args.in_path}")
    print(f"in_filetype: {args.in_filetype}")
    print(f"out_dir_path: {args.out_dir_path}")
    print(f"out_filetype: {args.out_filetype}")

    # Optional arguments
    if args.retain_dir_tree:
        print("retain_dir_tree: True")
    else:
        print("retain_dir_tree: False")
    if args.source_labels_filepath:
        print(f"source_labels_filepath: {args.source_labels_filepath}")
    if args.target_model_filepath:
        print(f"target_model_filepath: {args.target_model_filepath}")
    if args.target_labels_filepath:
        print(f"target_labels_filepath: {args.target_labels_filepath}")
    if args.use_ensemble:
        print("use_ensemble: True")
    else:
        print("use_ensemble: False")
    if args.ensemble_class_model_selections:
        print(f"ensemble_class_model_selections: {args.ensemble_class_model_selections}")
    if args.min_confidence:
        print(f"min_confidence: {args.min_confidence}")
    if args.retain_logit_score:
        print("retain_logit_score: True")
    else:
        print("retain_logit_score: False")
    if args.n_processes:
        print(f"n_processes: {args.n_processes}")
    if args.n_separation:
        print(f"n_separation: {args.n_separation}")
    if args.digits:
        print(f"digits: {args.digits}")
    
    # print('process_file_or_dir')
    process_audio.process_file_or_dir(
        in_path                         = args.in_path,
        in_filetype                     = args.in_filetype,
        out_dir_path                    = args.out_dir_path,
        retain_dir_tree                 = args.retain_dir_tree,
        target_model_filepath           = args.target_model_filepath,
        source_labels_filepath          = args.source_labels_filepath,
        target_labels_filepath          = args.target_labels_filepath,
        use_ensemble                    = args.use_ensemble,
        ensemble_class_model_selections = args.ensemble_class_model_selections,
        min_confidence                  = args.min_confidence,
        retain_logit_score              = args.retain_logit_score,
        n_processes                     = args.n_processes,
        num_separation                  = args.n_separation,
        cleanup                         = True,
        sort_by                         = ['start_time', 'confidence'],
        ascending                       = [True, False],
        out_filetype                    = args.out_filetype,
        digits                          = args.digits
    )
    
    print('run_process_audio_script END')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script with required and optional arguments")

    # Required arguments
    parser.add_argument("in_path",      type=str, help="Absolute path to a single audio file or a directory containing audio files (will search the directory tree recursively)")
    parser.add_argument("in_filetype",  type=str, help="Supported file types: '.wav', '.aif', '.flac', '.mp3'")
    parser.add_argument("out_dir_path", type=str, help="Absolute path to the output directory")
    parser.add_argument("out_filetype", type=str, help="Supported file types: '.csv' (human-readable, larger storage size) or '.parquet' (compressed, smaller storage size)")

    # Optional arguments
    parser.add_argument("--retain_dir_tree", action="store_true", help="Flag to retain the original directory tree structure relative to input directory")
    parser.add_argument("--source_labels_filepath", type=str, help="Relative path to source model labels .txt file")
    parser.add_argument("--target_model_filepath", type=str, help="Relative path to target model .tflite file")
    parser.add_argument("--target_labels_filepath", type=str, help="Relative path to target model labels .txt file")
    parser.add_argument("--use_ensemble", action="store_true", help="Flag to use source and target models together as an ensemble")
    parser.add_argument("--ensemble_class_model_selections", type=str, help="Class-specific model selections for ensemble")
    parser.add_argument("--min_confidence", type=float, help="Minimum confidence score to retain a detection (float)")
    parser.add_argument("--retain_logit_score", action="store_true", help="Flag to retain raw logit scores from predictions")
    parser.add_argument("--n_processes", type=int, help="Number of cores used by the processing pool (<= number of physical cores available on your computer) (int)")
    parser.add_argument("--n_separation", type=int, help="Number of channels to separate for analysis. Leave as 1 to process original file alone (int)")
    parser.add_argument("--digits", type=int, help="Digits to round confidence scores (int)")

    args = parser.parse_args()
    # if len(sys.argv) > 1:
    #     args = parser.parse_args()
    # else:
    #     # Define default values
    #     args = parser.parse_args([
    #         "my_in_path",      # in_path
    #         "my_in_filetype",  # in_filetype
    #         "my_out_dir_path", # out_dir_path
    #         "my_out_filetype", # out_filetype
    #         "--target_model_filepath", "my_custom_model_filepath",  # optional argument
    #         "--retain_dir_tree"                       # optional flag (boolean)
    #     ])

    main(args)