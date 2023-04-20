## Calculate acoustic indices for an hour-long recording and plot alongside spectrogram
library(soundecology)
library(tuneR)
library(seewave)
library(tidyr)
source('helpers.R')

# Load test data
file = '~/Desktop/oesf-examples/04_30min.wav'
# file = '~/Desktop/oesf-examples/short/04_short.wav'

# Parameters
# BIO, ACI
min_freq = 2000
# BIO, ADI, AEI, ACI
max_freq = 8000
# ADI, AEI
freq_step    = 1000
db_threshold = -50
# ACI
j = 5
# NDSI
anthro_min = 0
anthro_max = 2000
bio_min    = 2000
bio_max    = 8000
# FFT, ACI, NDSI
window = 512 # FFT window size

interval = 60  # acoustic index sample length (sec)

sound = readWave(file)
if (dc_offset(sound@left)) sound@left = dc_correct(sound@left)

results = data.frame()
  
dur = length(sound@left) / sound@samp.rate
  
start = 0.0
while (start < dur) {
  end = min(start + interval, dur)
  message(start, ' to ', end, '...')
  subsample = cutw(sound, f=sound@samp.rate, from=start, to=end, output='Wave')
  
  # Compute indices
  message('BIO')
  bio = bioacoustic_index(
    soundfile = subsample,
    min_freq  = min_freq,
    max_freq  = max_freq,
    fft_w     = window
  )
  message('ADI')
  adi = acoustic_diversity(
    soundfile = subsample,
    max_freq  = max_freq,
    freq_step = freq_step,
    db_threshold = db_threshold,
    shannon = TRUE
  )
  message('AEI')
  aei = acoustic_evenness(
    soundfile = subsample,
    max_freq  = max_freq,
    freq_step = freq_step,
    db_threshold = db_threshold
  )
  message('M')
  m = M(
    wave = subsample
  )
  message('H')
  h = H(
    wave = subsample,
    wl = window
  )
  message('ACI')
  aci = acoustic_complexity(
    soundfile = subsample,
    min_freq  = min_freq,
    max_freq  = max_freq,
    fft_w     = window,
    j         = j
  )
  message('NDSI')
  ndsi = ndsi(
    soundfile = subsample,
    fft_w = window,
    anthro_min = anthro_min,
    anthro_max = anthro_max,
    bio_min = bio_min,
    bio_max = bio_max
  )
  # Store results
  results = rbind(results, data.frame(
    end = end,
    BIO = bio$left_area,
    ADI = adi$adi_left,
    AEI = aei$aei_left,
    M   = m,
    H   = h,
    ACI = aci$AciTotAll_left,
    NDSI = ndsi$ndsi_left
  ))
  start = end
}

# Normalize
minmaxnorm = function(x){(x-min(x))/(max(x)-min(x))}
results$ACI = minmaxnorm(results$ACI)
results$BIO = minmaxnorm(results$BIO)
results$ADI = sapply(results$ADI, function(x){(x-0)/(log(10)-0)}) # natural log of 10 bins
results$AEI = sapply(results$AEI, function(x){(x-0)/(log(10)-0)})
results$M   = results$M/max(results$M)
results$NDSI = sapply(results$NDSI, function(x){(x-0)/(1.0-0.0)})

# Plot results
# library(patchwork)
ggplot(gather(results, 'index', 'value', c('BIO','ADI','AEI', 'M','H','ACI','NDSI')), aes(x=end, y=value, color=index)) + geom_line() + theme_minimal()

spectrogram(sound, alim=c(db_threshold,0), interpolate = F, wl = window) +
  theme_minimal()

# print(p_sp / p_ai)

