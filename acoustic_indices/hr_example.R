## Calculate acoustic indices for an hour-long recording and plot alongside spectrogram
library(soundecology)
library(tuneR)
library(seewave)
source('helpers.R')

# Load test data
file = '~/Desktop/oesf-examples/04.wav'

# Parameters
min_freq = 2000 # Hz
max_freq = 8000
window   = 512  # FFT window size

sound = readWave(file)
if (dc_offset(sound@left)) sound@left = dc_correct(sound@left)

# Calculate index
bio = bioacoustic_index(
  soundfile = sound,
  min_freq  = min_freq, # default 2000
  max_freq  = max_freq, # default 8000
  fft_w     = window
)

# Plot
# p = spectrogram(sound,
#                 alim = c(-55,0),
#                 flim = c(0,12),
#                 tlim = c(2000,3600),
#                 interpolate = TRUE) +
#   theme_minimal()
# print(p)
