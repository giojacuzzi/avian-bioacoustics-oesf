## Exploratory sequence for manually tuning acoustic index parameters
library(seewave)

source('acoustic_indices/batch_process_alpha_indices.R')
source('global.R')
source('helpers.R')

file = '~/Desktop/oesf-examples/tuning/SMA00310_20200528_050047.wav'
wav = readWave(file)
interval = 60 # Take first 60 seconds of file
wav = cutw(wav, f=wav@samp.rate, from=0, to=interval, output='Wave')
wav@left = dc_correct(wav@left)

min_freq = 2000
max_freq = 9000
db_threshold = -45

spectrogram(wav, alim = c(db_threshold, 0)) + 
  annotate('rect', xmin = 0, xmax = interval, ymin = min_freq/1000, ymax = max_freq/1000,
           alpha = .1, color = 'white', fill = 'white')
