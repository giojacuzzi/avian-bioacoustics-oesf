# Load required packages -------------------------------------------------
from annotation.annotations import *
from utils.log import *

# Evaluate all species classes...
species_to_evaluate = 'all'
# ...or look at just a few
species_to_evaluate = ["pacific-slope flycatcher", "evening grosbeak"]
# species_to_evaluate = ["varied thrush", "pacific wren", "evening grosbeak", "hairy woodpecker", "golden-crowned kinglet", "barred owl", "chestnut-backed chickadee", "macgillivray's warbler", "townsend's warbler"]

# Get raw annotation data
raw_annotations = get_raw_annotations()

species = labels.get_species_classes()
if species_to_evaluate == 'all':
    species_to_evaluate = sorted(species)

# Collate raw annotation data into species detection labels per species
collated_detection_labels = collate_annotations_as_detections(raw_annotations, species_to_evaluate, only_annotated=False, print_detections=True)

print(collated_detection_labels.to_string())