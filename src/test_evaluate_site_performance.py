# NOTE: Must run test_evaluate_performance.py first with 'test' evaluation dataset

import pandas as pd
from utils.log import *
from utils.files import *
from classification.performance import *
import sys

overwrite = False
threshold = 0.1

custom_model_stub  = 'custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0'
out_dir = f'data/test/{custom_model_stub}'
out_dir_pretrained = out_dir + '/pre-trained'
out_dir_custom     = out_dir + '/custom'
models = [out_dir_custom] # [out_dir_pretrained, out_dir_custom]

class_labels_csv_path = os.path.abspath(f'data/class_labels.csv')
class_labels = pd.read_csv(class_labels_csv_path)
preexisting_labels_to_evaluate = list(class_labels[class_labels['novel'] == 0]['label_birdnet'])
novel_labels_to_evaluate = list(class_labels[class_labels['novel'] == 1]['label_birdnet'])
target_labels_to_evaluate = list(class_labels[class_labels['train'] == 1]['label_birdnet'])

print(f'{len(preexisting_labels_to_evaluate)} preexisting labels to evaluate:')
print(preexisting_labels_to_evaluate)
# input()

perf_metrics_and_thresholds = pd.read_csv('/Users/giojacuzzi/Downloads/performance_metrics.csv')

# Data culling – get minimum confidence score to retain a prediction for analysis (helps speed up analysis process considerably)
# intersection = [item for item in target_labels_to_evaluate if item in preexisting_labels_to_evaluate]
labels_to_evaluate = [label.split('_')[1].lower() for label in preexisting_labels_to_evaluate]
# species_class_thresholds = perf_metrics_and_thresholds[perf_metrics_and_thresholds['label'].isin(labels_to_evaluate)]
class_thresholds = perf_metrics_and_thresholds
threshold_min_Tp  = min(class_thresholds['Tp'])
threshold_min_Tf1 = min(class_thresholds['Tf1'])
class_thresholds['min'] = class_thresholds.apply(lambda row: min(row['Tp'], row['Tf1'], 0.5), axis=1)
class_thresholds.loc[class_thresholds['min'] < 0.01, 'min'] = 0.5 # set missing classes to 0.5 minimum
print(class_thresholds[['label', 'Tp', 'Tf1', 'min']].to_string())
min_conf_dict = dict(zip(class_thresholds['label'], class_thresholds['min']))

# sys.exit()

def remove_extension(f):
    return os.path.splitext(f)[0]

# TODO: Generate model prediction scores (source and target) for ALL raw audio files

# Load analyzer prediction scores for ALL raw audio files
if overwrite:
    for model in models:
        print(f'Loading {model} prediction scores for test examples...')

        if model == out_dir_pretrained:
            score_dir_root = f'data/test/raw_predictions/pretrained'
        elif model == out_dir_custom:
            score_dir_root = f'data/test/raw_predictions/{custom_model_stub}'

        score_files = []
        score_files.extend(find_files(score_dir_root, '.csv')) 
        predictions = pd.DataFrame()
        i = 0
        for file in score_files:
            if i % 50 == 0:
                print(f"{round(i/len(score_files) * 100, 2)}% ({i} of {len(score_files)} files)")
            score = pd.read_csv(file)
            if not {'common_name', 'confidence'}.issubset(score.columns):
                print_warning(f'Incompatible columns in file {file}. Skipping...')
                continue
            score.drop(columns=['start_date'], inplace=True)
            score['common_name'] = score['common_name'].str.lower()

            # TODO: Cull unnecessary predictions below relevant confidence thresholds
            # print('before')
            # print(score)
            score = score[
                score.apply(lambda row: row['confidence'] >= min_conf_dict.get(row['common_name'], float('-inf')), axis=1)
            ]
            # print('after')
            # print(score)

            score['file'] = os.path.basename(file)
            predictions = pd.concat([predictions, score], ignore_index=True)
            i += 1
        predictions['file'] = predictions['file'].apply(remove_extension)
        # predictions.rename(columns={'file': 'file'}, inplace=True)
        predictions.rename(columns={'common_name': 'label_predicted'}, inplace=True)
        predictions['label_predicted'] = predictions['label_predicted'].str.lower()

        if model == out_dir_pretrained:
            predictions.to_csv('/Users/giojacuzzi/Downloads/predictions_pretrained.csv', index=False)
        elif model == out_dir_custom:
            predictions_custom = predictions
            predictions.to_csv('/Users/giojacuzzi/Downloads/predictions_custom.csv', index=False)

predictions_pretrained = pd.read_csv('/Users/giojacuzzi/Downloads/predictions_pretrained.csv')
predictions_custom = pd.read_csv('/Users/giojacuzzi/Downloads/predictions_custom.csv')

# DEBUG
print(predictions_custom)
# input()

print(f'PERFORMANCE EVALUATION - site level ================================================================================================')

# Load site true presence and absence
print('Loading site true presence and absence...')
site_presence_absence = pd.read_csv('data/test/site_presence_absence.csv', header=None)

print('Site key:')
site_key = pd.read_csv('data/site_key.csv')
site_key['date_start'] = pd.to_datetime(site_key['date_start'], format='%Y%m%d').dt.date
site_key['date_end'] = pd.to_datetime(site_key['date_end'], format='%Y%m%d').dt.date
# site_key = site_presence_absence.iloc[:5].reset_index(drop=True)
# new_columns = site_key.iloc[:, 0].tolist()
# site_key = site_key.transpose()
# site_key = site_key[1:]
# site_key.columns = new_columns
# site_key['months'] = site_key['months'].apply(lambda x: list(map(int, x.split(','))))
print(site_key)

site_presence_absence = site_presence_absence.iloc[3:].reset_index(drop=True)
site_presence_absence.set_index(0, inplace=True)
site_presence_absence.columns = site_key['site']
nan_rows = site_presence_absence[site_presence_absence.isna().any(axis=1)]  # Select rows with any NaN values
if not nan_rows.empty:
    print(f"WARNING: NaN values found. Dropping...")
    site_presence_absence = site_presence_absence.dropna()  # Drop rows with NaN values
print(site_presence_absence)

# Calculate true species richness at each site
print('True species richness')
def sum_list(x):
    numeric_column = pd.to_numeric(x, errors='coerce') # coerce ? to NaN
    return int(numeric_column.sum())
true_species_richness = site_presence_absence.apply(sum_list)
print(true_species_richness)

def within_date_range(d, start, end):
    return start.date() <= d.date() <= end.date()

def get_matching_site(row):
    # print(row)
    # print(site_key)
    # input()
    match = site_key[
        (site_key['serialno'] == row['serialno']) & 
        (site_key['date_start'] <= row['date'].date()) & 
        (site_key['date_end'] >= row['date'].date())
    ]
    if not match.empty:
        return match.iloc[0]['site']
    else:
        print_error(f'Could not find matching site for data {row}')
        return None

site_level_perf = pd.DataFrame()
site_level_perf_mean = pd.DataFrame()
for model in models:
    print(f'BEGIN MODEL EVALUATION {model} (site level) --------------------------------------------------------------------')

    if model == out_dir_pretrained:
        # Find matching unique site ID for each prediction
        cpp = predictions_pretrained.copy()
        model_labels_to_evaluate = [label.split('_')[1].lower() for label in preexisting_labels_to_evaluate]
    elif model == out_dir_custom:
        cpp = predictions_custom.copy()
        intersection = [item for item in target_labels_to_evaluate if item in preexisting_labels_to_evaluate]
        model_labels_to_evaluate = [label.split('_')[1].lower() for label in intersection]
    # cpp = cpp[cpp['confidence'] > 0.1]

    cpp['site'] = ''

    print('Calculating site-level performance metrics...')

    # Calculate perf with thresholds optimized for precision and F1 score
    print('Calculate site-level performance per label...')
    metrics = perf_metrics_and_thresholds[perf_metrics_and_thresholds['model'] == model]
    print('metrics')
    print(metrics)
    # input()
    # metrics_custom = performance_metrics[performance_metrics['model'] == out_dir_custom]
    # print('metrics_custom')
    # print(metrics_custom)
    
    counter = 1
    for label in ['pacific wren']: # model_labels_to_evaluate
        print(f'Evaluating site-level performance for class "{label}" ({counter})...')
        counter += 1
        # print(cpp)
        print('Copying relevant data...')
        predictions_for_label = cpp[cpp['label_predicted'] == label].copy()
        # input()
        print('Parsing metadata...')
        metadata = predictions_for_label['file'].apply(parse_metadata_from_detection_audio_filename)
        # print('METADATA')
        # print(metadata)
        # print(f"len(metadata) {len(metadata)}")
        # print(f"len(predictions_for_label) {len(predictions_for_label)}")
        serialnos = metadata.apply(lambda x: x[0]).tolist()
        dates = metadata.apply(lambda x: x[1]).tolist()
        times = metadata.apply(lambda x: x[2]).tolist()
        predictions_for_label['serialno'] = serialnos
        predictions_for_label['date']     = dates
        predictions_for_label['time']     = times

        # print('predictions_for_label unique date and serialno pairings:')
        # print(predictions_for_label[['date', 'serialno']].drop_duplicates().to_string())
        # input()
        predictions_for_label['date'] = pd.to_datetime(predictions_for_label['date'], format='%Y%m%d')
        # print_success('predictions_for_label')
        # print(predictions_for_label)
        # input()
        print('Retrieving site IDs...')
        for i, row in predictions_for_label.iterrows():
            # print(i)
            # print(row)
            serialno = row['serialno']
            date = row['date']
            site = get_matching_site(row)
            # if serialno == 'SMA00556':
            #     print(f"got site {site} for serialno {serialno} and date {date}")
            predictions_for_label.at[i, 'site'] = site
        print('PREDICTIONS')
        print(predictions_for_label)
        print('SITES')
        print(predictions_for_label['site'].unique())

        # DEBUG
        monkey = 'Bp236i'
        print(f'predictions_for_label {monkey}')
        print(predictions_for_label[predictions_for_label['site'] == monkey])
        input()

        # Pre-trained model
        # print('METRICS PRETRAINED')
        label_metrics = metrics[metrics['label'] == label]
        Tp = label_metrics['Tp'].iloc[0]
        Tf1 = label_metrics['Tf1'].iloc[0]

        threshold_labels = ['Tp', 'Tf1', '0.5', '0.9', 'max_Tp_0.5', 'max_Tp_0.9']
        thresholds       = [Tp, Tf1, 0.5, 0.9, max(Tp, 0.5), max(Tp, 0.9)]
        print('thresholds')
        print(threshold_labels)
        print(thresholds)
        # input()

        species_perf = pd.DataFrame()
        for i, threshold in enumerate(thresholds):
            threshold_label = threshold_labels[i]
            # print(f'Calculating site-level confusion matrix with {threshold_label} threshold {threshold}...')

            species_perf_at_threshold = get_site_level_confusion_matrix(label, predictions_for_label, threshold, site_presence_absence)
            species_perf_at_threshold['model'] = model
            species_perf_at_threshold['threshold'] = threshold_label
            species_perf = pd.concat([species_perf, species_perf_at_threshold], ignore_index=True)

        print(species_perf.to_string())
        site_level_perf = pd.concat([site_level_perf, species_perf], ignore_index=True)

    print(f'FINAL RESULTS {model} (site level) ------------------------------------------------------------------------------------------------------')
    print(site_level_perf.to_string())
    site_level_perf.to_csv('/Users/giojacuzzi/Downloads/site_level_perf.csv', index=False)
    input()

    # site_level_perf = site_level_perf.dropna() # remove any species for which there is no valid data (e.g. species without a corresponding optimized threshold)

print('DEBUG')
print(f'pretrained N={len(site_level_perf[site_level_perf["model"] == out_dir_pretrained])/len(threshold_labels)}')
print(f'custom N={len(site_level_perf[site_level_perf["model"] == out_dir_custom])/len(threshold_labels)}')
# input()

print('SITE LEVEL PERF COMPARISON ==================================================================================================')

labels_to_compare = [l for l in target_labels_to_evaluate if l in preexisting_labels_to_evaluate]
labels_to_compare = [l.split('_')[1].lower() for l in labels_to_compare]
print('labels_to_compare')
print(labels_to_compare)
input()

for threshold_label in threshold_labels:
    print_warning(f'>> Evaluating site-level performance for {threshold_label}...')

    threshold_results = pd.DataFrame()
    for model in models:
        print(f'Evaluating model {model}...')

        # Get the results matching the model and the threshold, model_results
        model_results = site_level_perf[(site_level_perf['threshold'] == threshold_label) & (site_level_perf['model'] == model)].copy()
        model_results = model_results[model_results['label'].isin(labels_to_compare)]
        print('model_results')
        print_exclude_cols = ['sites_detected', 'sites_notdetected', 'sites_error']
        print(model_results.drop(columns=print_exclude_cols).to_string())
        print(f"mean error %: {model_results['error_pcnt'].mean()}")
        print(f"min error %: {model_results['error_pcnt'].min()}")
        print(f"max error %: {model_results['error_pcnt'].max()}")
        print(f"mean precision: {model_results['precision'].mean()}")
        print(f"mean recall: {model_results['recall'].mean()}")
        print(f"mean fpr: {model_results['fpr'].mean()}")
        # input()
        threshold_results = pd.concat([threshold_results,model_results[['label', 'error_pcnt', 'precision', 'recall', 'fpr', 'model']]], ignore_index=True)

    # print_success(threshold_results.to_string())

    merged = pd.merge(threshold_results[threshold_results['model'] == out_dir_pretrained], threshold_results[threshold_results['model'] == out_dir_custom], on='label', suffixes=('_pretrained', '_custom'))
    merged['error_pcnt_Δ'] = merged['error_pcnt_custom'] - merged['error_pcnt_pretrained']
    merged['precision_Δ']  = merged['precision_custom'] - merged['precision_pretrained']
    merged['recall_Δ']     = merged['recall_custom'] - merged['recall_pretrained']
    merged['fpr_Δ']     = merged['fpr_custom'] - merged['fpr_pretrained']
    result = merged[['label', 'error_pcnt_Δ', 'precision_Δ', 'recall_Δ', 'fpr_Δ']]
    # print_warning(merged)
    print_success(result)
    print_success(f'mean error_pcnt_Δ {result["error_pcnt_Δ"].mean()}')
    print_success(f'mean precision_Δ  {result["precision_Δ"].mean()}')
    print_success(f'mean recall_Δ     {result["recall_Δ"].mean()}')
    print_success(f'mean fpr_Δ        {result["fpr_Δ"].mean()}')
    input()


# SPECIES RICHNESS COMPARISON
sys.exit()
# print('SPECIES RICHNESS COMPARISON ==================================================================================================')

# # For each threshold
# for threshold_label in threshold_labels:
#     print_warning(f'>> Evaluating site-level performance for {threshold_label}...')

#     # For each model
#     for model in models:
#         print(f'Evaluating model {model}...')

#         if model == out_dir_pretrained:
#             m = 'pretrained'
#         elif model == out_dir_custom:
#             m = 'custom'
#         print(m)

#         # Get the results matching the model and the threshold, model_results
#         model_results = site_level_perf[(site_level_perf['threshold'] == threshold_label) & (site_level_perf['model'] == model)].copy()
#         print('model_results')
#         print_exclude_cols = ['sites_detected', 'sites_notdetected', 'sites_error']
#         print(model_results.drop(columns=print_exclude_cols).to_string())
#         input()

#         # If the model under evaluation is custom...
#         if model == out_dir_custom:
#             print('CUSTOM MODEL!')
#             input()
            
#             # Get the results pretrained_results matching the pre-trained model and threshold
#             pretrained_results = site_level_perf[(site_level_perf['threshold'] == threshold_label) & (site_level_perf['model'] == out_dir_pretrained)].copy()
#             print('pretrained_results')
#             print(pretrained_results)
#             input()

#             # Replace all rows in model_results with label values NOT in the trained list with the rows for those labels in pretrained_results
#             labels_to_replace = [l for l in preexisting_labels_to_evaluate if l not in target_labels_to_evaluate]
#             labels_to_replace = [l.split('_')[1].lower() for l in labels_to_replace]
#             print('labels_to_replace')
#             print(labels_to_replace)
#             pretrained_results = pretrained_results[pretrained_results['label'].isin(labels_to_replace)]
#             print('pretrained_results')
#             print(pretrained_results)
#             model_results = model_results[~model_results['label'].isin(labels_to_replace)]
#             print('model_results')
#             print(model_results)
#             model_results = pd.concat([model_results, pretrained_results], ignore_index=True)
#             print('combined')
#             print(model_results)
#             input()
        
#         # Calculate stats and store for later, compute stat deltas between models for each threshold
#         print('Site species counts:') # Species richness comparison
#         df_exploded = model_results.explode('sites_detected') # make one row per site-species detection
#         site_species_counts = df_exploded.groupby('sites_detected')['label'].count() # get count of species (i.e. labels) for each site
#         site_species_counts = site_species_counts.reset_index(name='species_count')

#         site_species_counts['true_species_richness'] = site_species_counts['sites_detected'].map(true_species_richness)
#         site_species_counts['delta'] = site_species_counts['species_count'] - site_species_counts['true_species_richness']
#         site_species_counts['delta_pcnt'] = (site_species_counts['species_count'] / site_species_counts['true_species_richness']) * 100.0

#         # TODO: add error_pcnt

#         # Display the updated DataFrame
#         print(site_species_counts)
#         input()

#         print(f"Total average site precision: {model_results['precision'].mean()}")
#         print(f"Total average site recall: {model_results['recall'].mean()}")
#         print(f"Total average site error percentage: {model_results['error_pcnt'].mean()}")
#         print(f"Total average species richness percentage: {site_species_counts['delta_pcnt'].mean()}")
#         mean_site_perf_at_threshold = pd.DataFrame({
#             "precision":  [model_results['precision'].mean()], # total average site precision
#             "recall":     [model_results['recall'].mean()], # total average site recall
#             "error_pcnt": [model_results['error_pcnt'].mean()], # total average site error %
#             "delta_pcnt": [site_species_counts['delta_pcnt'].mean()], # total average species richness % of truth
#             "threshold":  [threshold_label],
#             "model":      [model]
#         })
#         print(mean_site_perf_at_threshold)
#         site_level_perf_mean = pd.concat([site_level_perf_mean, mean_site_perf_at_threshold], ignore_index=True)
#         input()

#         # Determine effect of habitat type on performance
#         print('Average species richness percentage Δerence by strata:')
#         merged_df = pd.merge(site_key, site_species_counts, left_on='site', right_on='sites_detected', how='inner')
#         print('merged_df')
#         print(merged_df)
#         average_percentage_Δ_by_stratum = merged_df.groupby('stratum')['delta_pcnt'].mean()
#         print('Species richness Δerence:')
#         print(average_percentage_Δ_by_stratum)
#         # TODO:
#         # print('Site error Δerence:')
#         # average_error_Δ_by_stratum = merged_df.groupby('stratum')['error_pcnt'].mean()
#         # print(average_error_Δ_by_stratum)
#         input()

#         # TODO: Determine effect of vocal activity on site-level performance vs. detection-level performance (do very frequent vocalizers perform better at the site level?)
#         # - Correlation between detection-level performance and site-level performance
#         # - Correlation between vocal activity (number of true examples in the test dataset) and site-level performance

#         # TODO: Compare number of detections per label between models using an optimized threshold

# print('FINAL MEAN SITE LEVEL PERF:')
# print_success(site_level_perf_mean.to_string())
