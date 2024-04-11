from utils import files
from utils.log import *

# Run the routine
raw_metadata = files.get_raw_metadata(
    [
        '/Volumes/gioj_b1/OESF/2020'
    ],
    overwrite=True)

print(raw_metadata)