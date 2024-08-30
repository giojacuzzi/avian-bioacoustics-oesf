# Calculate performance from k-fold cross validation results
# NOTE: Run test_compare_validation_performance.py on all required models first
overwrite = True

import os
import pandas as pd
import sys

custom_models_stub = 'custom_S1_N100_A0_U0_I' # CHANGE ME

desired_metrics = ['AUC-PR', 'AUC-ROC', 'f1_max'] # AP?
round_digits = 3

def list_dirs_with_substring(directory, substring):
    matching_dirs = []
    for item in os.listdir(directory):
        if item == substring:
            continue
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path) and item.startswith(substring):
            matching_dirs.append(item)
    return matching_dirs

# Calculate custom and pre-trained model cross-validation metrics
custom_model_metrics     = pd.DataFrame()
pretrained_model_metrics = pd.DataFrame()
iterations = sorted(list_dirs_with_substring('data/validation/custom', custom_models_stub))
for custom_model_stub in iterations:
    print(custom_model_stub)
    custom_model_dir_path = f'data/validation/custom/{custom_model_stub}'

    custom_model_iteration_metrics = pd.read_csv(f'{custom_model_dir_path}/metrics_custom.csv', index_col=0)
    custom_model_metrics = pd.concat([custom_model_metrics, custom_model_iteration_metrics], ignore_index=True)

    pretrained_model_iteration_metrics = pd.read_csv(f'{custom_model_dir_path}/metrics_pre-trained.csv', index_col=0)
    pretrained_model_metrics = pd.concat([pretrained_model_metrics, pretrained_model_iteration_metrics], ignore_index=True)

print('Custom model metrics:')
print(custom_model_metrics.to_string())
print('Pretrained model metrics:')
print(pretrained_model_metrics.to_string())

def get_xvalidation_metrics(model_metrics, tag=''):
    mean_values_per_label = model_metrics.groupby('label')[desired_metrics].mean(numeric_only=True)
    mean_values_per_label['metric'] = 'mean'
    mean_values = pd.DataFrame([mean_values_per_label.mean(numeric_only=True)], index=['COMBINED'])
    mean_values['metric'] = 'mean'

    sd_values_per_label = model_metrics.groupby('label')[desired_metrics].std(numeric_only=True)
    sd_values_per_label['metric'] = 'sd'
    sd_values = pd.DataFrame([sd_values_per_label.std(numeric_only=True)], index=['COMBINED'])
    sd_values['metric'] = 'sd'

    model_xvalidation_metrics = pd.concat([mean_values, sd_values, mean_values_per_label, sd_values_per_label])
    model_xvalidation_metrics = model_xvalidation_metrics.sort_values(by=['metric']).sort_index()
    model_xvalidation_metrics['label'] = model_xvalidation_metrics.index
    model_xvalidation_metrics = model_xvalidation_metrics.rename(columns={col: col + tag if col not in ['label', 'metric'] else col for col in model_xvalidation_metrics.columns})
    return(model_xvalidation_metrics)

custom_xvalidation_metrics = get_xvalidation_metrics(custom_model_metrics, tag='_custom')
pretrained_xvalidation_metrics = get_xvalidation_metrics(pretrained_model_metrics, tag='_pretrained')

# print('CUSTOM MODEL CROSS-VALIDATION METRICS:')
# print(custom_xvalidation_metrics.head())
# print('PRE-TRAINED MODEL CROSS-VALIDATION METRICS:')
# print(pretrained_xvalidation_metrics.head())

# Calculate deltas between custom and pre-trained models 
combined_xvalidation_metrics = pd.merge(custom_xvalidation_metrics, pretrained_xvalidation_metrics, on=['label', 'metric'], how='left', suffixes=('_custom', '_pretrained'))
for column in desired_metrics:
    delta_column = f'{column}_Δ'
    combined_xvalidation_metrics[delta_column] = combined_xvalidation_metrics[f'{column}_custom'] - combined_xvalidation_metrics[f'{column}_pretrained']
xvalidation_delta_metrics = combined_xvalidation_metrics[['label', 'metric'] + [f'{col}_Δ' for col in desired_metrics]]
xvalidation_delta_metrics.index = xvalidation_delta_metrics['label']
xvalidation_delta_metrics.index.name = None

# print('DELTA CROSS-VALIDATION METRICS:')
# print(xvalidation_delta_metrics.head())

print('=========================================')

xvalidation_metrics = pd.merge(xvalidation_delta_metrics, custom_xvalidation_metrics, on=['label', 'metric'])
xvalidation_metrics = pd.merge(xvalidation_metrics, pretrained_xvalidation_metrics, on=['label', 'metric'])
xvalidation_metrics.index = xvalidation_metrics['label']

# Print
def output_formatted_metrics(model_metrics):
    model_xvalidation_metrics = pd.DataFrame()
    for label in sorted(model_metrics['label'].unique()):
        label_metrics = model_metrics[model_metrics['label'] == label]
        m = label_metrics[label_metrics['metric'] == 'mean'].round(round_digits)
        sd = label_metrics[label_metrics['metric'] == 'sd'].round(round_digits)
        row = pd.DataFrame(m.astype(str)) + ' (' + pd.DataFrame(sd.astype(str)) + ')'
        row.index = [label]
        model_xvalidation_metrics = pd.concat([model_xvalidation_metrics, row], axis=0)
    model_xvalidation_metrics = model_xvalidation_metrics.drop_duplicates()
    model_xvalidation_metrics = model_xvalidation_metrics.drop(['label', 'metric'], axis=1)
    model_xvalidation_metrics = model_xvalidation_metrics.sort_index(axis=1)
    model_xvalidation_metrics.index.name = 'label'
    print('mean (standard deviation)')
    print(model_xvalidation_metrics)

    out_path = f'data/validation/custom/{custom_models_stub}'
    os.makedirs(out_path, exist_ok=True)
    model_xvalidation_metrics.to_csv(f'{out_path}/xval_metrics_summary.csv')

output_formatted_metrics(xvalidation_metrics)
