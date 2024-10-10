import pandas as pd
import numpy as np
import ast

site_key = pd.read_csv('data/site_key.csv')
# print(site_key)

# Site-level stratum with the most site errors using minimum error threshold, both models

path_site_perf_pretrained = 'data/results/site_perf/pretrained/site_perf_pretrained.csv'
site_perf_pretrained = pd.read_csv(path_site_perf_pretrained)
site_perf_pretrained = site_perf_pretrained[site_perf_pretrained['threshold'] == '0.8']
print(site_perf_pretrained) 

path_site_perf_custom = 'data/results/site_perf/custom/site_perf_custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0.csv'
site_perf_custom = pd.read_csv(path_site_perf_custom)
site_perf_custom = site_perf_custom[site_perf_custom['threshold'] == '0.9']
print(site_perf_custom.to_string())

print(site_key)

def get_list_from_string(s):
    return(ast.literal_eval(s.replace(" ", ",")))

#####################################
for model in ['source_all', 'target_all', 'source', 'target']:

    print(f'Evaluating model {model}...')

    if model == 'source_all':
        site_perf = site_perf_pretrained.reset_index()
    elif model == 'target_all':
        site_perf = site_perf_pretrained[~site_perf_pretrained['label'].isin(site_perf_custom['label'])]
        site_perf = pd.concat([site_perf, site_perf_custom], ignore_index=True)
    elif model == 'source':
        site_perf = site_perf_pretrained[site_perf_pretrained['label'].isin(site_perf_custom['label'])].reset_index()
    elif model == 'target':
        site_perf = site_perf_custom.reset_index()
    
    print(site_perf['label'])
    # input()

    # For each species
    species_by_stratum_site_perf = pd.DataFrame()
    for index, row in site_perf.iterrows():
        label = row['label']
        # print(f'label {label}')
        sites_error = get_list_from_string(row['sites_error'])
        # print(f'sites_error {sites_error}')
        sites_valid = get_list_from_string(row['sites_valid'])
        # print(f'sites_valid {sites_valid}')
        sites_detected = get_list_from_string(row['sites_detected'])
        sites_notdetected = get_list_from_string(row['sites_notdetected'])

        # Get the sites_error by stratum
        # Get the sites_valid by stratum

        # input()

        # For each stratum
        temp_stratum_error_rates = pd.DataFrame()
        for stratum in site_key['stratum'].unique():

            # print(f'stratum {stratum}')
            # Calculate error rate for this species across sites of this stratum
            # sites_error of stratum / sites_valid of stratum

            stratum_sites = site_key[site_key['stratum'] == stratum]['site']

            sites_error_stratum = stratum_sites[stratum_sites.isin(sites_error)]
            # print(f'sites_error_stratum {sites_error_stratum.to_list()}')

            sites_valid_stratum = stratum_sites[stratum_sites.isin(sites_valid)]
            # print(f'sites_valid_stratum {sites_valid_stratum.to_list()}')

            sites_detected_stratum = stratum_sites[stratum_sites.isin(sites_detected)]
            sites_fp_stratum = sites_detected_stratum[sites_detected_stratum.isin(sites_error)]
            # print(f'sites_fp_stratum {sites_fp_stratum.to_list()}')

            sites_notdetected_stratum = stratum_sites[stratum_sites.isin(sites_notdetected)]
            sites_fn_stratum = sites_notdetected_stratum[sites_notdetected_stratum.isin(sites_error)]
            # print(f'sites_fn_stratum {sites_fn_stratum.to_list()}')
            # input()

            try:
                error_rate = len(sites_error_stratum) / len(sites_valid_stratum)
            except ZeroDivisionError:
                error_rate = np.nan
            
            try:
                fp_rate = len(sites_fp_stratum) / len(sites_valid_stratum)
            except ZeroDivisionError:
                fp_rate = np.nan
            
            try:
                fn_rate = len(sites_fn_stratum) / len(sites_valid_stratum)
            except ZeroDivisionError:
                fn_rate = np.nan

            # print(f'error_rate {error_rate}')

            # Save
            temp = pd.DataFrame([{
                'class': label,
                'stratum': stratum,
                'error_rate': error_rate,
                'fp_rate': fp_rate,
                'fn_rate': fn_rate
            }])
            # print(f'temp {temp}')
            
            # input()
            temp_stratum_error_rates = pd.concat([temp_stratum_error_rates, temp], ignore_index=True)
            # print(f'temp_stratum_error_rates {temp_stratum_error_rates}')
        
        species_by_stratum_site_perf = pd.concat([species_by_stratum_site_perf,temp_stratum_error_rates], ignore_index=True)

    print(f'species_by_stratum_error_rates')
    print(species_by_stratum_site_perf.to_string())

    print('ERROR RATE:')
    results_error_rates = species_by_stratum_site_perf.pivot(index='class', columns='stratum', values='error_rate')
    results_error_rates.reset_index(inplace=True)
    results_error_rates['model'] = model

    # print(results_error_rates.to_string())
    results_error_rates_means = results_error_rates.mean(numeric_only=True)
    print(f'model "{model}" results_error_rates_means:')
    print(results_error_rates_means.round(2))

    print('FALSE POSITIVE RATE:')
    results_fp_rates = species_by_stratum_site_perf.pivot(index='class', columns='stratum', values='fp_rate')
    results_fp_rates.reset_index(inplace=True)
    results_fp_rates['model'] = model

    # print(results_fp_rates.to_string())
    results_fp_rates_means = results_fp_rates.mean(numeric_only=True)
    print(f'model "{model}" results_fp_rates_means:')
    print(results_fp_rates_means.round(2))

    print('FALSE NEGATIVE RATE:')
    results_fn_rates = species_by_stratum_site_perf.pivot(index='class', columns='stratum', values='fn_rate')
    results_fn_rates.reset_index(inplace=True)
    results_fn_rates['model'] = model

    # print(results_fn_rates.to_string())
    results_fn_rates_means = results_fn_rates.mean(numeric_only=True)
    print(f'model "{model}" results_fn_rates_means:')
    print(results_fn_rates_means.round(2))

    input(f'Model {model} finished. Press [return] to continue...')

#####################################

# NOTE: THIS INTRODUCES BIAS BASED ON WHICH SITES HAVE MORE SPECIES AT THEM!
# def count_errors_by_stratum(site_perf):
#     site_counts = {}
#     for index, row in site_perf.iterrows():
#         # print(f"Index: {index}")
#         # print(f"Row:\n{row}")
#         sites_error = row['sites_error']
#         # print(sites_error)
#         sites_error = sites_error.replace(" ", ",")
#         # Convert to an actual Python list
#         # print(sites_error)
#         sites_error = ast.literal_eval(sites_error)
#         # print(sites_error)
#         for site in set(sites_error):
#             # print(site)
#             if site in site_counts:
#                 site_counts[site] += 1
#             else:
#                 site_counts[site] = 1
#     # print(site_counts)
#     df_site_counts = pd.DataFrame(list(site_counts.items()), columns=['site', 'num_error'])
#     # print(df_site_counts)

#     merged_df = pd.merge(site_key, df_site_counts, on='site', how='outer')  # Change 'inner' to 'outer', 'left', or 'right' as needed
#     print(merged_df)

#     total_errors = merged_df.groupby('stratum')['num_error'].sum().reset_index()
#     print(total_errors)

# print('Pre-trained errors by stratum:')
# count_errors_by_stratum(site_perf_pretrained)
# print('Custom errors by stratum:')
# count_errors_by_stratum(site_perf_custom)

# Sample-level PR AUC mean by strata, both models
# TODO




