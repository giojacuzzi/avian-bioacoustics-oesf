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
print(detections)
all_sites = detections["StationName"].unique()
ntotal_sites = len(all_sites)
print(f'{ntotal_sites} unique sites annotated')
if (ntotal_sites != 16):
    print_error('Missing sites?')
    sys.exit()

## 1. "Arbitrary" confidence thresholds (0.5, 0.9) versus an "informed" threshold to maximize precision ######################################
# > For each species:
#   - Number of sites detected / number of sites truly present
#   - Focus on focal species (Marbled Murrelet)
# > Estimated species richness delta across sites (versus "true" species richness)

# def get_site_level_confusion_matrix(species, detections, threshold):


# TODO: CHECK GREAT-HORNED OWL
for species in species_list:
    print(f'{species}')

    # Filter for species detections, excluding unknown examples
    detections_species = detections[(detections['label_predicted'] == species) & (detections['label_truth'] != 'unknown')]

    # print(detections_species.to_string())

    # Number of sites truly present (based on TP detections)
    sites_truly_present  = detections_species[(detections_species['label_truth'] == species)]["StationName"].unique()
    print(f'Sites truly present ({len(sites_truly_present)}): {sites_truly_present}')
    # print(detections_species[(detections_species['label_truth'] == species)])

    sites_truly_absent = np.setdiff1d(all_sites, sites_truly_present)
    print(f'Sites truly absent  ({ len(sites_truly_absent)}): {sites_truly_absent}')

    if len(sites_truly_present) + len(sites_truly_absent) != ntotal_sites:
        print_error(f'Number of sites truly present ({len(sites_truly_present)}) and absent ({len(sites_truly_absent)}) does not equal total number of sites ({ntotal_sites})')

    detections_0_5 = detections_species[(detections_species['confidence'] >= 0.5)]
    sites_detected_0_5 = detections_0_5["StationName"].unique()
    print(f'Sites detected 0.5  ({len(sites_detected_0_5)}): {sites_detected_0_5}')
    # print(detections_0_5)

    sites_not_detected_0_5 = np.setdiff1d(all_sites, sites_detected_0_5)
    print(f'Sites not detected 0.5  ({len(sites_not_detected_0_5)}): {sites_not_detected_0_5}')

    if len(sites_detected_0_5) + len(sites_not_detected_0_5) != ntotal_sites:
        print_error(f'Number of sites detected ({len(sites_detected_0_5)}) and not detected ({len(sites_not_detected_0_5)}) does not equal total number of sites ({ntotal_sites})')

    # TP - Number of sites correctly detected (at least once)
    tp_0_5 = np.intersect1d(sites_truly_present, sites_detected_0_5)
    nsites_tp_0_5 = len(tp_0_5)
    print(f'  TP 0.5 - {nsites_tp_0_5}')

    # FP - Number of sites incorrectly detected (i.e. not correctly detected at least once)
    fp_0_5 = np.setdiff1d(np.intersect1d(sites_truly_absent, sites_detected_0_5), tp_0_5) # remove sites where the species was otherwise correctly detected at least once
    nsites_fp_0_5 = len(fp_0_5)
    print(f'  FP 0.5 - {nsites_fp_0_5}')

    # TN - Number of sites correctly not detected
    nsites_tn_0_5 = len(np.intersect1d(sites_not_detected_0_5, sites_truly_absent))
    print(f'  TN 0.5 - {nsites_tn_0_5}')

    # FN - Number of sites incorrectly not detected
    nsites_fn_0_5 = len(np.intersect1d(sites_not_detected_0_5, sites_truly_present))
    print(f'  FN 0.5 - {nsites_fn_0_5}')

    nsites_accounted_for = nsites_tp_0_5 + nsites_fp_0_5 + nsites_tn_0_5 + nsites_fn_0_5
    if nsites_accounted_for != ntotal_sites:
        print_error(f'Only {nsites_accounted_for} sites accounted for of {ntotal_sites} total')
    
    if nsites_tp_0_5 + nsites_fn_0_5 != len(sites_truly_present):
        print_error(f'Incorrect true presences TP {nsites_tp_0_5} + FN {nsites_fn_0_5} != {len(sites_truly_present)}')
    if nsites_fp_0_5 + nsites_tn_0_5 != len(sites_truly_absent):
        print_error(f'Incorrect true absences FP {nsites_fp_0_5} + TN {nsites_tn_0_5} != {len(sites_truly_absent)}')

## 2. "Informed" threshold side effects ######################################################################################################
# > For each species:
#   - Number of sites not detected / number of sites truly present

# TODO

## 3. Pre-trained classifier with "informed" threshold versus custom classifier with "informed" threshold ####################################
#     - Number of sites detected / number of sites truly present
#     - Number of sites not detected / number of sites truly present
#     - Number of sites detected / number of sites truly absent

# TODO
