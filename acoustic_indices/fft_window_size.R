# NOTES:
# Pieretti et al 2011 used an FFT size of 512, and recording lengths of 5-30 sec for ACI
# Boelman et al 2007 used recording lengths of 8 min for BIO
# Bradfer-Lawrence et al 2019 used recording lengths of 1 and 10 min for all indices
#
# The fft window size (aka number of points per FFT, aka the number of output frequency bins of each FFT, as a power of 2) is also known as the block length.
# 
# The maximum frequency that can be determined by the FFT is the nyquist, which is the sampling rate / 2
# The FFT window duration (in seconds) is the size divided by the sampling rate
# The frequency resolution of each bin is equal to the sampling rate divided by the FFT size. Bins are equally spaced from 0 Hz up to the Nyquist frequency.
#
# A small window results in fast measurement repetitions with a coarse frequency resolution.
# A large window results in slower measuring repetitions with fine frequency resolution.
# You have to balance the two depending on the context (i.e. input data)

library(soundecology)
library(tuneR)
library(seewave)
source('helpers.R')

file = '~/Desktop/oesf-examples/04.wav'

# NOTES:
# - Both BIO and ACI scale (increase) with FFT size (when holding time measurement interval delta constant)
# - Relative ACI values will vary slightly with FFT size
windows = c(64,128,256,512,1024,2048) # FFT window length
delta = 60*2

results = data.frame()
for (window in windows) {
  message('FFT ', window)
  sound = readWave(file)
  if (dc_offset(sound@left)) sound@left = dc_correct(sound@left)
  
  dur = length(sound@left) / sound@samp.rate
  
  i = 0.0
  while (i < dur) {
    j = min(i + delta, dur)
    test = cutw(sound, f=sound@samp.rate, from=i, to=j, output='Wave')
    
    # # Compute bioacoustic index
    # bio = bioacoustic_index(test, fft_w = window)
    # index = bio$left_area
    
    # Compute acoustic complexity index
    aci = acoustic_complexity(test, fft_w = window)
    index = aci$AciTotAll_left / (delta/60) # normalize
    
    message(i, ' to ', j, ' sec: ', index)
    
    results = rbind(results, data.frame(
      window = window,
      start = i,
      end   = j,
      index = index
    ))
    i = j
  }
}
results$window = factor(results$window)

# Add a duplicate data point for the last observation
# of each window size to complete geom_step lines below 
last_rows = aggregate(. ~ window, results, tail, n = 1)
last_rows$start = last_rows$end
results = rbind(results, last_rows)

# Plot results per window size
ggplot(results, aes(x=start, y=index, color=window)) +
  geom_step(linewidth=1) +
  scale_color_viridis_d() +
  labs(title = 'Effect of FFT window on acoustic index',
       subtitle = 'Calculated from a recording of the onset of the dawn chorus',
       x = 'Time (sec)', y = 'Acoustic index', color = 'Window\nLength') +
  theme_minimal()
