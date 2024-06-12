# This script calculates the primary performance metrics from the applied example in the OESF
# Currently, it only uses the 16 sites that have been manually annotated

# Annotation directories to use as evaluation data
dirs = [
    '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/_annotator/2020',
]

import numpy as np
from annotation.annotations import *
from classification.performance import *
from utils.files import *
from utils.log import *

site_deployment_metadata = get_site_deployment_metadata(2020)
print(site_deployment_metadata)

species_list = sorted(labels.get_species_classes())
# species_list = [
#     "marbled murrelet",
#     "red-breasted nuthatch",
#     "pacific-slope flycatcher",
#     "pacific wren",
#     "great horned owl",
#     "northern saw-whet owl",
#     "northern pygmy-owl",
#     "varied thrush"
# ] # DEBUG

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
    detections_thresholded = detections_species[(detections_species['confidence'] >= threshold)]
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
    # try:
    #     tp_pcnt = round(nsites_tp / len(sites_present), 2)
    # except ZeroDivisionError:
    #     tp_pcnt = np.nan
    # try:
    #     tn_pcnt = round(nsites_tn / len(sites_absent), 2)
    # except ZeroDivisionError:
    #     tn_pcnt = np.nan
    
    result = {
        'species':        [species],
        'present':        [len(sites_present)],
        'absent':         [len(sites_absent)],
        'threshold':      [threshold],
        'detected':       [len(sites_detected)],
        'notdetected':    [len(sites_notdetected)],
        'correct':        [nsites_tp + nsites_tn],
        'correct_pcnt':   [round((nsites_tp + nsites_tn) / (len(sites_present) + len(sites_absent)), 2)],
        'error':          [nsites_fp + nsites_fn],
        'error_pcnt':     [round((nsites_fp + nsites_fn) / (len(sites_present) + len(sites_absent)), 2)],
        'FP': [nsites_fp],
        'FN': [nsites_fn],
        'TP': [nsites_tp],
        'TN': [nsites_tn],
        # 'TP_pcnt': [tp_pcnt],
        # 'TN_pcnt': [tn_pcnt],
        'precision': [round(precision,3)],
        'recall':    [round(recall,3)]
    }
    return(pd.DataFrame(result, index=None))

site_level_perf = pd.DataFrame()

## 1. "Arbitrary" confidence thresholds (0.5, 0.9) versus an "optimal" threshold to maximize precision ######################################

# > For each species:
#   - Number of sites detected / number of sites truly present
#   - Focus on focal species (Marbled Murrelet)
if True:
    for species in species_list:
        print(f'Calculating site-level performance metrics for species {species} with thresholds 0.5 and 0.9 ...')

        species_perf_0_5 = get_site_level_confusion_matrix(species, detections, 0.5)
        site_level_perf = pd.concat([site_level_perf, species_perf_0_5], ignore_index=True)

        species_perf_0_9 = get_site_level_confusion_matrix(species, detections, 0.9)
        site_level_perf = pd.concat([site_level_perf, species_perf_0_9], ignore_index=True)

    site_level_perf['optimization'] = np.nan
    print(site_level_perf.sort_values(by='error', ascending=False).to_string())

# Histogram of FP count for different naive thresholds (exclude sites with no likely presence)
precision_05 = site_level_perf[(site_level_perf['threshold'] == 0.5) & (site_level_perf['present'] > 1)]['precision']
recall_05 = site_level_perf[(site_level_perf['threshold'] == 0.5) & (site_level_perf['present'] > 1)]['recall']
fp_05 = site_level_perf[(site_level_perf['threshold'] == 0.5) & (site_level_perf['present'] > 1)]['FP']
fn_05 = site_level_perf[(site_level_perf['threshold'] == 0.5) & (site_level_perf['present'] > 1)]['FN']
precision_09 = site_level_perf[(site_level_perf['threshold'] == 0.9) & (site_level_perf['present'] > 1)]['precision']
recall_09 = site_level_perf[(site_level_perf['threshold'] == 0.9) & (site_level_perf['present'] > 1)]['recall']
fp_09 = site_level_perf[(site_level_perf['threshold'] == 0.9) & (site_level_perf['present'] > 1)]['FP']
fn_09 = site_level_perf[(site_level_perf['threshold'] == 0.9) & (site_level_perf['present'] > 1)]['FN']

print(f'Across {site_level_perf[(site_level_perf["present"] > 1)]["species"].nunique()} unique species...')
# print_warning(site_level_perf[(site_level_perf["threshold"] == 0.9) & (site_level_perf["present"] > 1)])
print(f'Mean precision 0.5 -> {round(precision_05.mean(),2)}')
print(f'Mean recall 0.5 -> {round(recall_05.mean(),2)}')
print(f'Mean FP 0.5 -> {round(fp_05.mean(),2)} ({round(fp_05.mean()/ntotal_sites * 100, 1)}% of sites)')
print(f'Mean FN 0.5 -> {round(fn_05.mean(),2)} ({round(fn_05.mean()/ntotal_sites * 100, 1)}% of sites)')
print(f'Mean precision 0.9 -> {round(precision_09.mean(),2)}')
print(f'Mean recall 0.9 -> {round(recall_09.mean(),2)}')
print(f'Mean FP 0.9 -> {round(fp_09.mean(),2)} ({round(fp_09.mean()/ntotal_sites * 100, 1)}% of sites)')
print(f'Mean FN 0.9 -> {round(fn_09.mean(),2)} ({round(fn_09.mean()/ntotal_sites * 100, 1)}% of sites)')

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# ax1.hist(fp_05, bins=range(0, max(fp_05.max(), fp_09.max()) + 2), alpha=0.5, label='Threshold = 0.5', color='blue')
# ax1.hist(fp_09, bins=range(0, max(fp_05.max(), fp_09.max()) + 2), alpha=0.5, label='Threshold = 0.9', color='red')
# ax1.set_xlabel('FP values')
# ax1.set_ylabel('Frequency')
# ax1.set_title('Histogram of FP values for naive thresholds')
# ax1.legend()
# ax2.hist(fn_05, bins=range(0, max(fn_05.max(), fn_09.max()) + 2), alpha=0.5, label='Threshold = 0.5', color='blue')
# ax2.hist(fn_09, bins=range(0, max(fn_05.max(), fn_09.max()) + 2), alpha=0.5, label='Threshold = 0.9', color='red')
# ax2.set_xlabel('FN values')
# ax2.set_ylabel('Frequency')
# ax2.set_title('Histogram of FN values for naive thresholds')
# ax2.legend()
# plt.tight_layout()
# plt.show()

# > Estimated species richness delta across sites (versus "true" species richness)
# TODO

## 2. "Optimal" threshold side effects ######################################################################################################
# > For each species:
#   - Number of sites not detected / number of sites truly present
if True:

    species_thresholds_cache_path = 'data/cache/species_thresholds.csv'
    if True:
        # Calculate threshold to maximize precision/recall for each species, and cache results for later
        species_level_perf = pd.DataFrame()
        for species in species_list:
            print(f'Calculating species-level performance metrics for species {species}...')

            detection_labels = collated_detection_labels[collated_detection_labels['label_predicted'] == species]
            species_performance_metrics = evaluate_species_performance(detection_labels, species, plot = False)
            # plt.show()
            species_level_perf = pd.concat([species_level_perf, species_performance_metrics], ignore_index=True)
        species_level_perf = species_level_perf.sort_values(by='p_max_th', ascending=False)
        species_level_perf.to_csv(species_thresholds_cache_path, index=False)
        print(species_level_perf.sort_values(by=['species'], ascending=True).to_string())
    species_level_perf = pd.read_csv(species_thresholds_cache_path)

    for species in species_list:
        print(f'Calculating site-level performance metrics for species {species} with threshold to maximize precision/recall...')

        if species not in species_level_perf['species'].values:
            print_warning(f'Skipping species with missing threshold {species}...')
            continue

        threshold_maxp = species_level_perf[species_level_perf['species'] == species]['p_max_th'].values[0]
        species_perf_maxp = get_site_level_confusion_matrix(species, detections, threshold_maxp)
        species_perf_maxp['optimization'] = 'precision'
        site_level_perf = pd.concat([site_level_perf, species_perf_maxp], ignore_index=True)

        threshold_maxr = species_level_perf[species_level_perf['species'] == species]['r_max_th'].values[0]
        species_perf_maxr = get_site_level_confusion_matrix(species, detections, threshold_maxr)
        species_perf_maxr['optimization'] = 'recall'
        site_level_perf = pd.concat([site_level_perf, species_perf_maxr], ignore_index=True)
    
    print(site_level_perf.sort_values(by=['species', 'threshold'], ascending=True).to_string())
    print_success(site_level_perf[(site_level_perf['present'] > 1)].sort_values(by=['species', 'threshold'], ascending=True).to_string())

# Also calculate mean FP/FN across all species when using their respective optimal thresholds for precision/recall
precision_maxp = site_level_perf[(site_level_perf['optimization'] == 'precision') & (site_level_perf['present'] > 1)]['precision']
recall_maxp = site_level_perf[(site_level_perf['optimization'] == 'precision') & (site_level_perf['present'] > 1)]['recall']
fp_maxp = site_level_perf[(site_level_perf['optimization'] == 'precision') & (site_level_perf['present'] > 1)]['FP']
fn_maxp = site_level_perf[(site_level_perf['optimization'] == 'precision') & (site_level_perf['present'] > 1)]['FN']
precision_maxr = site_level_perf[(site_level_perf['optimization'] == 'recall') & (site_level_perf['present'] > 1)]['precision']
recall_maxr = site_level_perf[(site_level_perf['optimization'] == 'recall') & (site_level_perf['present'] > 1)]['recall']
fp_maxr = site_level_perf[(site_level_perf['optimization'] == 'recall') & (site_level_perf['present'] > 1)]['FP']
fn_maxr = site_level_perf[(site_level_perf['optimization'] == 'recall') & (site_level_perf['present'] > 1)]['FN']

print(f'Across {site_level_perf[(site_level_perf["present"] > 1)]["species"].nunique()} unique species...')
print(f'Mean precision maxp -> {round(precision_maxp.mean(),2)}')
print(f'Mean recall maxp -> {round(recall_maxp.mean(),2)}')
print(f'Mean FP maxp -> {round(fp_maxp.mean(),2)} ({round(fp_maxp.mean()/ntotal_sites * 100, 1)}% of sites)')
print(f'Mean FN maxp -> {round(fn_maxp.mean(),2)} ({round(fn_maxp.mean()/ntotal_sites * 100, 1)}% of sites)')
print(f'Mean precision maxr -> {round(precision_maxr.mean(),2)}')
print(f'Mean recall maxr -> {round(recall_maxr.mean(),2)}')
print(f'Mean FP maxr -> {round(fp_maxr.mean(),2)} ({round(fp_maxr.mean()/ntotal_sites * 100, 1)}% of sites)')
print(f'Mean FN maxr -> {round(fn_maxr.mean(),2)} ({round(fn_maxr.mean()/ntotal_sites * 100, 1)}% of sites)')

# List species-level performance
print_success(f'\n{species_level_perf[(species_level_perf["N_P"] > 1)].sort_values(by=["AUC-PR"], ascending=True).to_string()}')

## 3. Pre-trained classifier with "optimal" threshold versus custom classifier with "optimal" threshold ####################################
#     - Number of sites detected / number of sites truly present
#     - Number of sites not detected / number of sites truly present
#     - Number of sites detected / number of sites truly absent

# TODO