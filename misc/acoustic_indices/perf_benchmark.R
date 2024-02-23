## Compare performance of acoustic index functions
library(soundecology)
library(tuneR)
library(seewave)
library(tidyr)
source('helpers.R')

# Load test data
file = '~/Desktop/oesf-examples/short/04_short.wav'

# Parameters
# BIO, ACI
min_freq = 2000 # Hz
# BIO, ADI, AEI, ACI
max_freq = 8000
# ADI, AEI
freq_step    = 1000
db_threshold = -40  # Note: this threshold can be used to prevent the influence of microphone self-noise
# ACI
j = 5 # cluster size (sec)
# FFT, ACI
window   = 512 # FFT window size
interval = 60*2   # acoustic index sample length (sec)

sound = readWave(file)
if (dc_offset(sound@left)) sound@left = dc_correct(sound@left)

library(microbenchmark)

results = microbenchmark(
  bio = bioacoustic_index(
    soundfile = sound,
    min_freq  = min_freq,
    max_freq  = max_freq,
    fft_w     = window
  ),
  adi = acoustic_diversity(
    soundfile = sound,
    max_freq  = max_freq,
    freq_step = freq_step,
    db_threshold = db_threshold,
    shannon = TRUE
  ),
  aei = acoustic_evenness(
    soundfile = sound,
    max_freq  = max_freq,
    freq_step = freq_step,
    db_threshold = db_threshold
  ),
  th = th(
    env = env(sound)
  ),
  sh = sh(
    spec = spec(sound)
  ),
  m = M(
    wave = sound
  ),
  h = H(
    wave = sound,
    wl = window
  ),
  aci = acoustic_complexity(
    soundfile = sound,
    min_freq  = min_freq,
    max_freq  = max_freq,
    fft_w     = window,
    j         = j
  ),
  times = 50)
