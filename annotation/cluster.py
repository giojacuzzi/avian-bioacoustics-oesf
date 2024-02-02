# path = '/Users/giojacuzzi/Desktop/audio_test/owl.wav'
path = '/Users/giojacuzzi/Desktop/audio_test/chorus.wav'
sep = True # perform source separation
num_sources = 4
db_max=100  # define spectrogram range (i.e. db threshold)

import classification.sound_separation as sound_separation
import os
import numpy as np
import matplotlib.pyplot as plt
from maad import sound, features, rois
from maad.spl import power2dBSPL
from maad.util import power2dB, plot2d, format_features, overlay_rois
from maad.util import rand_cmap
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import subprocess
import sys
import pandas as pd

print(os.path.dirname(path))

if sep:
    # Separate audio into 4 sources
    files = sound_separation.separate(path, num_sources)
    file_mix = ([file for file in files if 'source' not in file])[0]
    files = [file for file in files if 'source' in file] # discard original mix
    print(files)
else:
    files = [path]
    file_mix = path

all_rois = pd.DataFrame()
all_X = pd.DataFrame()

# Use scikit-maad to find ROIs for each source
for f in files:

    # Load source as spectrogram
    print(f'Loading {f} as spectrogram...')
    s, fs = sound.load(f)
    s = sound.select_bandwidth(s, fs, fcut=500, forder=1, ftype='highpass')
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=512)
    Sxx_db = power2dB(Sxx, db_range=db_max) + db_max
    # plot2d(Sxx_db, **{'extent':ext})
    # plt.show()
    # print(tn)
    # print(fn)

    # Find regions of interest
    print('Finding regions of interest...')
    Sxx_db_rmbg, _, _ = sound.remove_background(Sxx_db)
    Sxx_db_smooth = sound.smooth(Sxx_db_rmbg, std=1.2)
    im_mask = rois.create_mask(im=Sxx_db_smooth, mode_bin ='relative', bin_std=2, bin_per=0.25)
    im_rois, df_rois = rois.select_rois(im_mask, min_roi=500, max_roi=None)
    
    # Format ROIs and visualize the bounding box on the audio spectrogram.
    df_rois = format_features(df_rois, tn, fn)
    print(f"Found {len(df_rois)} ROIs")
    if len(df_rois) == 0:
        continue

    # ax0, fig0 = overlay_rois(Sxx_db, df_rois, **{'vmin':0, 'vmax':60, 'extent':ext})
    # plt.show()

    all_rois = pd.concat([all_rois, df_rois], ignore_index=True)

    # Compute acoustic features
    print('Computing acoustic features...')
    df_shape, params = features.shape_features(Sxx_db, resolution='low', rois=df_rois)
    df_centroid = features.centroid_features(Sxx_db, df_rois)

    # Get median frequency and normalize
    median_freq = fn[np.round(df_centroid.centroid_y).astype(int)]
    df_centroid['centroid_freq'] = median_freq/fn[-1]

    X = df_shape.loc[:,df_shape.columns.str.startswith('shp')]
    X = X.join(df_centroid.centroid_freq) # add column and normalize values
    all_X = pd.concat([all_X, X], ignore_index=True)

all_rois['labelID'] = all_rois.index

tsne = TSNE(n_components=2, perplexity=min(len(all_rois) - 1, 12), init='pca', verbose=True)
Y = tsne.fit_transform(all_X)

# fig, ax = plt.subplots()
# ax.scatter(Y[:,0], Y[:,1], c='gray', alpha=0.8)
# ax.set_xlabel('tsne dim 1')
# ax.set_ylabel('tsne dim 2')
# plt.show()

# Cluster ROIs into homogeneous groups
from sklearn.cluster import DBSCAN
cluster = DBSCAN(eps=3.5, min_samples=4).fit(Y)
print('Number of clusters found:', np.unique(cluster.labels_).size)

# from maad.util import rand_cmap
# fig, ax = plt.subplots()
# ax.scatter(Y[:,0], Y[:,1], c=cluster.labels_, cmap=rand_cmap(5 , first_color_black=False), alpha=0.8)
# ax.set_xlabel('tsne dim 1')
# ax.set_ylabel('tsne dim 2')

# Load mix as spectrogram
print(f'Loading {file_mix} as spectrogram...')
s, fs = sound.load(file_mix)
# s_filt = sound.select_bandwidth(s, fs, fcut=1000, forder=2, ftype='highpass')
Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=512)
Sxx_db = power2dB(Sxx, db_range=db_max) + db_max

# Overlay bounding box on the original spectrogram
all_rois['label'] = cluster.labels_.astype(str)
ax0, fig0 = overlay_rois(Sxx_db, all_rois, **{'vmin':0, 'vmax':60, 'extent':ext})
plt.show()


# Save clustered ROIs as a Raven Pro selection table
selection_table = all_rois.rename(columns={
    'labelID': 'Selection',
    'min_t': 'Begin Time (s)',
    'max_t': 'End Time (s)',
    'min_f': 'Low Freq (Hz)',
    'max_f': 'High Freq (Hz)',
    'label': 'species'
})
selection_table['Selection'] = (selection_table['Selection'].astype(int)) + 1
selection_table['View'] = 'Spectrogram 1'
selection_table['Channel'] = 1
selection_table = selection_table[['Selection','View','Channel','Begin Time (s)','End Time (s)','Low Freq (Hz)','High Freq (Hz)','species']]

print(selection_table)
outpath = f'{os.path.dirname(path)}/{os.path.splitext(os.path.basename(path))[0]}.Table.1.selections.txt'
print(outpath)
selection_table.to_csv(outpath, sep='\t', index=False)

# Open folder in finder
subprocess.run(['/usr/bin/open', outpath])
