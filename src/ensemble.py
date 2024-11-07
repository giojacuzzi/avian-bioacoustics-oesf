target_model  = 'custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0'

import pandas as pd
import numpy as np

# SAMPLE

file_source_perf = f'data/results/{target_model}/sample_perf/metrics_pre-trained.csv'
file_target_perf = f'data/results/{target_model}/sample_perf/metrics_custom.csv'

perf_source = pd.read_csv(file_source_perf)
perf_source['label'] = perf_source['label'].str.lower()
print(file_source_perf)
# print(perf_source.to_string())

perf_target = pd.read_csv(file_target_perf)
perf_target['label'] = perf_target['label'].str.lower()
print(file_target_perf)
# print(perf_target.to_string())

sample_perf_combined = pd.merge(
    perf_source[['label', 'PR_AUC']].rename(columns={'PR_AUC': 'PR_AUC_source'}),
    perf_target[['label', 'PR_AUC']].rename(columns={'PR_AUC': 'PR_AUC_target'}),
    on='label', how='outer'
)
sample_perf_combined['PR_AUC_max'] = sample_perf_combined[['PR_AUC_source', 'PR_AUC_target']].max(axis=1)
sample_perf_combined['PR_AUC_max_model'] = np.where(
    sample_perf_combined['PR_AUC_source'] == sample_perf_combined['PR_AUC_max'], 'source',
    np.where(sample_perf_combined['PR_AUC_target'] == sample_perf_combined['PR_AUC_max'], 'target', 'source')
)
print(sample_perf_combined.to_string())

# perf_combined.to_csv(f'data/results/{target_model}/sample_perf/metrics_combined.csv', index=False)

# SITE

threshold = str(0.9)

file_source_perf = f'data/results/{target_model}/site_perf/pretrained/site_perf_pretrained.csv' 
file_target_perf = f'data/results/{target_model}/site_perf/custom/site_perf_{target_model}.csv'

perf_source = pd.read_csv(file_source_perf)
perf_source['label'] = perf_source['label'].str.lower()
perf_source = perf_source[perf_source['threshold'] == threshold]

perf_target = pd.read_csv(file_target_perf)
perf_target['label'] = perf_target['label'].str.lower()
perf_target = perf_target[perf_target['threshold'] == threshold]
perf_target = perf_target[perf_target['present'] > 0]

site_perf_combined = pd.merge(
    perf_source[['label', 'correct_pcnt']].rename(columns={'correct_pcnt': f'accuracy_source_{threshold}'}),
    perf_target[['label', 'correct_pcnt']].rename(columns={'correct_pcnt': f'accuracy_target_{threshold}'}),
    on='label', how='outer'
)
site_perf_combined[f'accuracy_max_{threshold}'] = site_perf_combined[[f'accuracy_source_{threshold}', f'accuracy_target_{threshold}']].max(axis=1)
site_perf_combined[f'accuracy_max_{threshold}_model'] = np.where(
    site_perf_combined[f'accuracy_source_{threshold}'] == site_perf_combined[f'accuracy_max_{threshold}'], 'source',
    np.where(site_perf_combined[f'accuracy_target_{threshold}'] == site_perf_combined[f'accuracy_max_{threshold}'], 'target', 'source')
)
print(site_perf_combined.to_string())

# perf_combined.to_csv(f'data/results/{target_model}/site_perf/site_metrics_combined.csv', index=False)

combined = pd.merge(
    sample_perf_combined,
    site_perf_combined,
    on='label', how='outer'
)
print(combined.to_string())

combined.to_csv('/Users/giojacuzzi/Downloads/combined.csv', index=False)