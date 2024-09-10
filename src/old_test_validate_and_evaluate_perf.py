##
# Demo script to evaluate classifier performance for each species class
# Requires annotations from Google Drive. Ensure these are downloaded locally before running.
#
# For a given audio segment, the classifier outputs a logistic regression score for each
# species class representing its confidence that the species is present in the segment.
# We need to transform this continuous confidence score into a discrete binary classification
# (e.g. Robin present vs Robin not present). We can do this with a classification threshold.
#
# TODO: Load all raw classifier detections. This is needed in future for evaluating alternate classifiers one-to-one, or to use non-species-specific detections for evaluation
# TODO: Determine the most commonly-confused (associated) labels per species for false positives.

# Configuration variables ------------------------------------

# Evaluate all species classes...
species_to_evaluate = 'all'
species_wadnr_priority = ["pileated woodpecker", "pacific-slope flycatcher", "hutton's vireo", "chestnut-backed chickadee", "bewick's wren", "pacific wren", "varied thrush", "brown creeper", "orange-crowned warbler", "wilson's warbler", "spotted owl", "barred owl", "marbled murrelet", "canada jay", "steller's jay", "american crow", "common raven", "golden eagle", "northern goshawk", "peregrine falcon", "vaux's swift", "rufous hummingbird"]

# # ...or look at just a few
# species_to_evaluate = []
# # WADNR TARGET SPECIES
# species_to_evaluate += species_wadnr_priority
# # OTHER TARGETS
# species_to_evaluate += ["hermit thrush"]
# # INDIVIDUAL
# species_to_evaluate = ["wilson's warbler","pacific-slope flycatcher","marbled murrelet", "varied thrush", "northern saw-whet owl", "northern pygmy-owl", "white-crowned sparrow"]

plot = False              # Plot the results
print_detections = False # Print detections
sort_by = 'confidence' # Detection sort, if printing, e.g. 'confidence' or 'label_truth'

# TODO: Once all annotations are complete and every detection has been evaluated, set this to False
only_annotated = False # DEBUG: skip files that do not have annotations (selection tables)

# Annotation directories to use as evaluation data
dirs = [
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/transfer learning/data/test/2020',
]

import pandas as pd
import matplotlib.pyplot as plt
from annotation.annotations import *
from utils.log import *
from classification.performance import *
from utils.files import *

# Retrieve and validate raw annotation data
raw_annotations = get_raw_annotations(dirs = dirs, overwrite = True)

# Check for missing annotations
species = labels.get_species_classes()
if species_to_evaluate == 'all':
    species_to_evaluate = sorted(species)
    # Warn if missing annotations for any species in label_predicted
    species_annotated = [label for label in sorted(raw_annotations['label_truth'].astype(str).unique()) if label in species]
    print(f'Retrieved {len(raw_annotations)} total annotations for {len(species_annotated)}/{len(species)} species classes')
    if len(species_annotated) < len(species):
        print_warning(f'Missing positive annotations for species: {[label for label in species if label not in species_annotated]}')
else:
    species_to_evaluate = sorted(species_to_evaluate)

# Collate raw annotation data into species detection labels per species
collated_detection_labels = collate_annotations_as_detections(raw_annotations, species_to_evaluate, only_annotated=only_annotated)
collated_detection_labels['date'] = pd.to_datetime(collated_detection_labels['date'], format='%Y%m%d')

# Merge site deployment metadata
site_deployment_metadata = get_site_deployment_metadata(2020)
merged_df = pd.merge(
    collated_detection_labels, 
    site_deployment_metadata[['SerialNo', 'SurveyDate', 'StationName_AGG', 'Stratum']],
    left_on=['serialno', 'date'], 
    right_on=['SerialNo', 'SurveyDate'],
    how='left'
)
merged_df = merged_df.drop(columns=['SerialNo', 'SurveyDate']).drop_duplicates()
collated_detection_labels = merged_df
collated_detection_labels = collated_detection_labels.rename(columns={"StationName_AGG": "site", "Stratum": "stratum"})

# Print detections sorted by descending confidence
if print_detections:
    collated_detection_labels = collated_detection_labels.sort_values(by=[sort_by], ascending=[False])
    print(collated_detection_labels.to_string())

    pd.set_option('display.max_colwidth', None)
    print(f'Confirmed presence ({species_to_evaluate}):')
    print(collated_detection_labels[collated_detection_labels['label_truth'].isin(species_to_evaluate)]['file'].to_string(index=False))
    print('Unknown presence/absence ("unknown"):')
    print(collated_detection_labels[collated_detection_labels['label_truth'] == 'unknown']['file'].to_string(index=False))
    print('Confirmed absence ("0"):')
    print(collated_detection_labels[collated_detection_labels['label_truth'] == '0']['file'].to_string(index=False))

# Collate Macaulay reference detections per species
# collated_macaulay_references = collate_macaulay_references(species_to_collate=species_to_evaluate, print_detections=False)

# Containers for performance metrics of all species
performance_metrics = pd.DataFrame()
# macaulay_reference_performance_metrics = pd.DataFrame()

# Calculate performance metrics for each species
for i, species in enumerate(species_to_evaluate):

    print(f'Calculating performance metrics for "{species}"...')

    detection_labels = collated_detection_labels[collated_detection_labels['label_predicted'] == species]
    species_performance_metrics = evaluate_species_performance(detection_labels, species, plot, save_to_dir='data/cache/test_validate_and_evaluate_perf')
    performance_metrics = pd.concat([performance_metrics, species_performance_metrics], ignore_index=True)

    # macaulay_references = collated_macaulay_references[collated_macaulay_references['label_predicted'] == species]
    # macaulay_performance_metrics = evaluate_species_performance(macaulay_references, species, plot=False)
    # macaulay_reference_performance_metrics = pd.concat([macaulay_reference_performance_metrics, macaulay_performance_metrics], ignore_index=True)

# Sort and print the performance metrics
performance_metrics = performance_metrics.sort_values(by='p_mean', ascending=False).reset_index(drop=True)

max_confidence_per_species = raw_annotations.rename(columns={'label_predicted': 'label'}).groupby('label')['confidence'].max()
performance_metrics = pd.merge(performance_metrics, max_confidence_per_species, on='label', how='left')
performance_metrics = performance_metrics.rename(columns={'confidence': 'max_conf'})
potential_species = pd.read_csv(files.species_list_filepath, index_col=None, usecols=['common_name', 'rarity'])
potential_species = potential_species.rename(columns={'common_name': 'label'})
potential_species['label'] = potential_species['label'].str.lower()
performance_metrics = pd.merge(potential_species, performance_metrics, on='label', how='left')

cols_as_ints = ['N','N_P','N_N','N_unknown']
performance_metrics[cols_as_ints] = performance_metrics[cols_as_ints].fillna(0)
performance_metrics[cols_as_ints] = performance_metrics[cols_as_ints].astype(int)

if species_to_evaluate != 'all':
    performance_metrics = performance_metrics[performance_metrics['label'].isin(species_to_evaluate)]

performance_metrics = performance_metrics.sort_values(by=['AUC-PR', 'p_max_r'], ascending=[True, True])

# Export performance data for hierarchical edge bundling plot in R
label_perf_filepath = 'data/annotations/processed/label_perf_metrics.csv'
performance_metrics.to_csv(label_perf_filepath, index=False)
print_success(f'Saved label performance metrics to {label_perf_filepath}')

print('WADNR PRIORITY SPECIES =======================================================================================================')
wadnr_species = performance_metrics[performance_metrics['label'].isin(species_wadnr_priority)]
print(wadnr_species.to_string(index=False))

print('SPECIES CLASS PERFORMANCE =============================================================================================')
min_N_P = 1
missing_species = performance_metrics.loc[performance_metrics['N_P']<min_N_P]

performance_metrics = performance_metrics.loc[performance_metrics['N_P']>0]

print(performance_metrics.to_string(index=False))

print('COMMON SPECIES ========================================================================================================')
common_species = performance_metrics.loc[performance_metrics['rarity']=='C']
common_species = common_species.sort_values(by=['N_P', 'N_unknown', 'max_conf'], ascending=[False, False, False])
print(common_species.to_string(index=False))

print('RARE SPECIES ==========================================================================================================')
rare_species = performance_metrics.loc[performance_metrics['rarity']=='R']
rare_species = rare_species.sort_values(by=['N_P', 'N_unknown', 'max_conf'], ascending=[False, False, False])
print(rare_species.to_string(index=False))

print('MISSING SPECIES =======================================================================================================')
print_warning(f'{len(missing_species)} species with less than {min_N_P} positive examples:\n{missing_species.to_string(index=False)}')

# print('MACAULAY REFERENCE PERFORMANCE ========================================================================================')
# macaulay_reference_performance_metrics = macaulay_reference_performance_metrics.sort_values(by=['p_max_th'], ascending=[False])
# print(macaulay_reference_performance_metrics.to_string(index=False))

if plot:
    plt.show()

