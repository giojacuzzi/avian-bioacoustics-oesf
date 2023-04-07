# Bioacoustic Index (BI)
# Original: https://esajournals.onlinelibrary.wiley.com/doi/full/10.1890/07-0004.1
# Guidelines: https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13254

library(seewave)
data(tico)
data(orni)

library(tuneR)
tuneR::setWavPlayer('/usr/bin/afplay')

play(tico)
oscillo(tico)
spectro(tico)
spec(tico)

play(orni)
oscillo(orni)
spectro(orni)
spec(orni)

min_freq = 2000
max_freq = 8000
fft_w    = 512

# soundecology: “bioacoustic_index”
library(soundecology)
bi_soundecology_tico = bioacoustic_index(
  soundfile = tico,     # Wave loaded with readWave from tuneR package
  min_freq  = min_freq, # minimum frequency (Hz, default 2000)
  max_freq  = max_freq, # maximum frequency (Hz, default 8000)
  fft_w     = fft_w     # window size (default 512)
)

bi_soundecology_orni = bioacoustic_index(
  soundfile = orni, min_freq = min_freq, max_freq= max_freq, fft_w = fft_w
)

mean_dB = function(x, level="IL")
{
  if(level=="IL") {a <- 10} else {a <- 20}
  return(a*log10(mean(10^(x/a))))
}



# Manual (mono)
# https://github.com/ljvillanueva/soundecology/blob/master/soundecology/R/bioacoust_index.R

# Get sample rate and nyquist
samplingrate = soundfile@samp.rate
nyquist_freq <- samplingrate/2

#Get left channel
soundfile = tico
left<-channel(soundfile, which = c("left"))

# Compute 2D spectrogram of time wave (short-term fourier transform)
# Here, use 'max0' for maximum dB value at 0
spec_left <- spectro(left, f = samplingrate, wl = fft_w, plot = T,
                     dB = "max0")

# $amp returns the matrix corresponding to the amplitude values. Each column is a Fourier transform of length wl/2
spec_left_amp  = spec_left$amp
spec_left_time = spec_left$time
spec_left_freq = spec_left$freq

# Get average in time
# TODO: is this correct? compare graph with spec(left)
specA_left <- apply(spec_left_amp, 1, mean_dB)

library(ggplot2)
ggplot() + geom_line(data = data.frame(
  Amplitude=specA_left,
  Frequency=spec_left_freq
), aes(x=Frequency, y=Amplitude)) + theme_minimal()

# How much Hz are covered per row
# TODO: is this a correct calculation? does it align with spec_left_freq?
rows_width = length(specA_left) / nyquist_freq

min_row = min_freq * rows_width
max_row = max_freq * rows_width

#Select rows
specA_left_segment <- specA_left[min_row:max_row]
freq_range <- max_freq - min_freq
freqs <- seq(from = min_freq, to = max_freq, length.out = length(specA_left_segment))

specA_left_segment_normalized <- specA_left_segment - min(specA_left_segment)

#left_area <- trapz(freqs, specA_left_segment_normalized)
left_area <- sum(specA_left_segment_normalized * rows_width)

#########
# scikit maad: (Python)
# https://scikit-maad.github.io/generated/maad.features.bioacoustics_index.html

# Wildlife Acoustics
# Measures the area under the log amplitude spectrum curve in dB*kHz with the minimum dB level set to zero.
# - Designed as a proxy for bird abundance
# - Used as proxy for richness and abundance (eg Shamon et al 2021)
# - Measures total amount of acoustic energy
# - OUTPUT: 0 to infinity

# 
# Calculation details: Computes the area under each curve, including all frequency bands from 2 to 11 kHz with a dB value greater than the minimum dB value for each curve.
# Interpretation: A combination of sound intensity and frequency bands occupied. Designed to reflect biophony. Low values indicate little to no acoustic activity.

# https://github.com/patriceguyot/Acoustic_Indices

# "calculated as the area under the curve of the mean amplitude spectrum between two frequency limits, is a function of both the sound level and the number of frequency bands.""

# A function of both amplitude and number of occupied frequency bands between 2 and 11 kHz. Value is relative to the quietest 1 kHz frequency band; higher values indicate greater disparity between loudest and quietest bands... In Bradfer-Lawrence study, highest values produced by blanket cicada noise, with high amplitude and minimal variation among frequency bands. Low values arise when there is no sound between 2 and 11 kHz, although there is sometimes insect biophony outside these bounds.
# https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13254

