import pandas as pd
label = "sooty grouse"
threshold = 0.90
model_tag = 'custom'

print('Loading custom predictions from cache...')
predictions_for_label = pd.read_parquet(f'data/cache/predictions_for_label_{label}_{model_tag}.parquet')
predictions_for_label = predictions_for_label[predictions_for_label['label_predicted'] == label]
predictions_for_label = predictions_for_label[predictions_for_label['confidence'] >= threshold]

sites = sorted(predictions_for_label['site'].unique())
print('Unique sites:')
print(sites)

print('Predictions...')
for s in sites:
    print(s)
    print(predictions_for_label[predictions_for_label['site'] == s])