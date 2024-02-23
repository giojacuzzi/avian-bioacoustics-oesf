# Analyze a single audio file

# NOTE: Custom edits to birdnetlib are used here (see analyzer.py).
# Specifically, APPLY_SIGMOID flag set to False throughout to
# return logits, not sigmoid activations

from process_file import process_file
from tools import *
import os

out_dir = '/Users/giojacuzzi/Downloads' # Output directory for detection dataframe
sort_by = 'confidence'                  # Column to sort dataframe by
ascending = False                       # Column sort direction

# Analyzer config
min_confidence = 0.1 # Minimum confidence score to retain a detection
num_separation = 1   # Number of sounds to separate for analysis. Leave as 1 for original file alone.
cleanup = False      # Keep or remove any temporary files created through analysis

# -----------------------------------------------------------------------------

# Prompt the user for a file path
in_filepath = os.path.normpath(input('\033[34mDrag and drop file to analyze (requires full path): \033[0m'))[1:-1]
root_dir = os.path.dirname(in_filepath)

result = process_file(
    filepath = in_filepath,
    out_dir = out_dir,
    min_confidence=min_confidence,
    num_separation=num_separation,
    cleanup=cleanup,
    root_dir=root_dir,
    sort_by=sort_by,
    ascending=ascending
)
print(f'Finished analyzing {in_filepath}:')

if not result.empty:
    print_success(f'{len(result)} detections of {len(result["common_name"].unique())} unique species:')
    print(result)
else:
    print_warning('No detections')
