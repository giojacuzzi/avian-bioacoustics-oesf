# Analyze a single audio file

# NOTE: Custom edits to birdnetlib are used here (see analyzer.py).
# Specifically, APPLY_SIGMOID flag set to False throughout to
# return logits, not sigmoid activations

from classification import process_file
from utils.log import *
import os

# Output config
sort_by      = 'confidence' # Column to sort dataframe by
ascending    = False        # Column sort direction
save_to_file = False        # Save output to a file
out_dir      = ''           # Output directory (e.g. '/Users/giojacuzzi/Downloads'), if saving output to file

# Analyzer config
min_confidence = 0.1  # Minimum confidence score to retain a detection
num_separation = 1    # Number of sounds to separate for analysis. Leave as 1 for original file alone.
cleanup        = True # Keep or remove any temporary files created through analysis

# -----------------------------------------------------------------------------

# Prompt the user for a file path. Path must not contain single quotes (').
in_filepath = input('\033[34mDrag and drop file to analyze (requires full path): \033[0m')
root_dir = os.path.dirname(in_filepath)

# Normalize file paths to support both mac and windows
if in_filepath.startswith("'") and in_filepath.endswith("'"):
    in_filepath = in_filepath[1:-1]
in_filepath = os.path.normpath(in_filepath)
out_dir     = os.path.normpath(out_dir)

result = process_file.process_file(
    in_filepath    = in_filepath,
    out_dir        = out_dir,
    min_confidence = min_confidence,
    num_separation = num_separation,
    cleanup        = cleanup,
    sort_by        = sort_by,
    ascending      = ascending,
    save_to_file   = save_to_file
)

if result is not None:
    print_success(f'{len(result)} detections of {len(result["common_name"].unique())} unique species:')
    print(result.to_string())
else:
    print_warning('No detections')
