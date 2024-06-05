# This script calculates the primary performance metrics from the applied example in the OESF
# Currently, it only uses the 16 sites that have been manually annotated

# Annotation directories to use as evaluation data
dirs = [
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020',
]

import numpy as np
from annotation.annotations import *
from utils.files import *
from utils.log import *

site_deployment_metadata = get_site_deployment_metadata(2020)
print(site_deployment_metadata)

species_list = sorted(labels.get_species_classes())

# Retrieve and validate raw annotation data, then collate raw annotation data into species detection labels per species
raw_annotations = get_raw_annotations(dirs = dirs, overwrite = False)
collated_detection_labels = collate_annotations_as_detections(raw_annotations, species_list)
collated_detection_labels['date'] = pd.to_datetime(collated_detection_labels['date'], format='%Y%m%d')
print(collated_detection_labels)

detections = pd.merge(
    site_deployment_metadata, 
    collated_detection_labels, 
    left_on=['SerialNo', 'SurveyDate'], 
    right_on=['serialno', 'date'],
    how='inner'
)
all_sites = detections["StationName"].unique()
ntotal_sites = len(all_sites)
print(f'{ntotal_sites} unique sites annotated')
if (ntotal_sites != 16):
    print_error('Missing sites')
    sys.exit()

# Returns a dataframe containing a confusion matrix (TP, FP, FN, TN) and number of truly present/absent sites for a given species from a detection history
def get_site_level_confusion_matrix(species, detections, threshold):
    # Filter for species detections, excluding unknown examples
    detections_species = detections[(detections['label_predicted'] == species) & (detections['label_truth'] != 'unknown')]

    # Sites truly present (based on TP detections)
    sites_present  = detections_species[(detections_species['label_truth'] == species)]["StationName"].unique()
    # print(f'Sites present ({len(sites_present)}): {sites_present}')

    # Sites truly absent
    sites_absent = np.setdiff1d(all_sites, sites_present)
    # print(f'Sites absent  ({ len(sites_absent)}): {sites_absent}')

    if len(sites_present) + len(sites_absent) != ntotal_sites:
        print_error(f'Number of sites present ({len(sites_present)}) and absent ({len(sites_absent)}) does not equal total number of sites ({ntotal_sites})')

    # Sites detected using the threshold
    detections_thresholded = detections_species[(detections_species['confidence'] >= 0.5)]
    sites_detected = detections_thresholded["StationName"].unique()
    # print(f'Sites detected with threshold {threshold}  ({len(sites_detected)}): {sites_detected}')

    # Sites not detected using the threshold
    sites_notdetected = np.setdiff1d(all_sites, sites_detected)
    # print(f'Sites not detected with threshold {threshold}  ({len(sites_notdetected)}): {sites_notdetected}')

    if len(sites_detected) + len(sites_notdetected) != ntotal_sites:
        print_error(f'Number of sites detected ({len(sites_detected)}) and not detected ({len(sites_notdetected)}) does not equal total number of sites ({ntotal_sites})')

    # TP - Number of sites correctly detected (at least once)
    tp_sites = np.intersect1d(sites_present, sites_detected)
    nsites_tp = len(tp_sites)

    # FP - Number of sites incorrectly detected (i.e. not correctly detected at least once)
    fp_sites = np.setdiff1d(np.intersect1d(sites_absent, sites_detected), tp_sites) # remove sites where the species was otherwise correctly detected at least once
    nsites_fp = len(fp_sites)

    # TN - Number of sites correctly not detected
    nsites_tn = len(np.intersect1d(sites_notdetected, sites_absent))

    # FN - Number of sites incorrectly not detected
    nsites_fn = len(np.intersect1d(sites_notdetected, sites_present))

    nsites_accounted_for = nsites_tp + nsites_fp + nsites_tn + nsites_fn
    if nsites_accounted_for != ntotal_sites:
        print_error(f'Only {nsites_accounted_for} sites accounted for of {ntotal_sites} total')
    
    if nsites_tp + nsites_fn != len(sites_present):
        print_error(f'Incorrect true presences TP {nsites_tp} + FN {nsites_fn} != {len(sites_present)}')
    if nsites_fp + nsites_tn != len(sites_absent):
        print_error(f'Incorrect true absences FP {nsites_fp} + TN {nsites_tn} != {len(sites_absent)}')
    
    try:
        precision = nsites_tp / (nsites_tp + nsites_fp)
    except ZeroDivisionError:
        precision = np.nan
    try:
        recall = nsites_tp / (nsites_tp + nsites_fn)
    except ZeroDivisionError:
        recall = np.nan
    
    result = {
        'species':        [species],
        'present':        [len(sites_present)],
        'absent':         [len(sites_absent)],
        'threshold':      [threshold],
        'detected':       [len(sites_detected)],
        'notdetected':    [len(sites_notdetected)],
        'incorrect':      [nsites_fp + nsites_fn],
        'FP': [nsites_fp],
        'FN': [nsites_fn],
        'TP': [nsites_tp],
        'TN': [nsites_tn],
        'precision': [precision],
        'recall':    [recall]
    }
    return(pd.DataFrame(result, index=None))

## 1. "Arbitrary" confidence thresholds (0.5, 0.9) versus an "informed" threshold to maximize precision ######################################

# > For each species:
#   - Number of sites detected / number of sites truly present
#   - Focus on focal species (Marbled Murrelet)
perf = pd.DataFrame()
for species in species_list:
    print(f'Processing species {species}...')

    species_perf_0_5 = get_site_level_confusion_matrix(species, detections, 0.5)
    perf = pd.concat([perf, species_perf_0_5], ignore_index=True)

    species_perf_0_9 = get_site_level_confusion_matrix(species, detections, 0.9)
    perf = pd.concat([perf, species_perf_0_9], ignore_index=True)

print(perf.sort_values(by='incorrect', ascending=False).to_string())

# > Estimated species richness delta across sites (versus "true" species richness)

## 2. "Informed" threshold side effects ######################################################################################################
# > For each species:
#   - Number of sites not detected / number of sites truly present

# TODO

## 3. Pre-trained classifier with "informed" threshold versus custom classifier with "informed" threshold ####################################
#     - Number of sites detected / number of sites truly present
#     - Number of sites not detected / number of sites truly present
#     - Number of sites detected / number of sites truly absent

# TODO
