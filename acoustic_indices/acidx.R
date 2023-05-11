# Optimized acoustic index functions

# TODO:
# 1-minute intervals
# FFT window size of 512
# BIO:
# min_freq = 1700, max_freq = 10000
# ACI:
# min_freq = 1700, max_freq = , j = 5

source('global.R')
library(tuneR)
library(seewave)
library(soundecology)
library(vegan)
library(ineq)

################################################################################
# Bioacoustic Index (BIO)
# Only compatible with single-channel (mono) data
# Output identical to `bioacoustic_index` from soundecology package
acidx_bio = function(
    data,          # Either a tuneR Wave object or a `spectro` spectrogram
    fs = 0,        # Sampling rate (hz)
    wl = 512,      # FFT window length
    fmin = 2000,   # Minimum frequency (hz)
    fmax = 8000,   # Maximum frequency (hz)
    ...            # Additional parameters for internal `spectro` call, if `data` is a Wave object. Note that dB=NULL is internally specified, as the max dB at zero conversion is performed by this function
) {
  if (class(data)=='Wave') # an audio file was provided
    data = spectro(channel(data, which='left'), fs, plot=F, dB=NULL, ...)

  spectrum = apply(20*log10(data$amp), 1, meandB) # average in time
  
  # hz per row
  rows_width = length(spectrum) / (fs/2.0) # nyquist
  min_row = fmin * rows_width
  max_row = fmax * rows_width

  spectrum = spectrum[min_row:max_row] # subset spectrum to [min,max]
  spectrum_normalized = spectrum - min(spectrum)
  
  bio = sum(spectrum_normalized * rows_width) # area
  return(bio)
}

################################################################################
# Acoustic Complexity Index (ACI)
# Only compatible with single-channel (mono) data
# Note that the result is the cumulative ACI total divided by the duration of the
# data, in minutes, to facilitate comparison of values computed from different durations
# When wn='hamming' is specified, output is identical to that of
# `soundecology::acoustic_complexity()`, divided by the number of minutes
acidx_aci = function(
    data,          # Either a tuneR Wave object or a `spectro` spectrogram
    fs = 0,        # Sampling rate (hz)
    wl = 512,      # FFT window length
    fmin = 2000,   # Minimum frequency (hz)
    fmax = 8000,   # Maximum frequency (hz)
    j = 5,         # Cluster size (sec)
    ...            # Additional parameters for internal `spectro` call, if `data` is a Wave object
) {
  if (class(data)=='Wave') # an audio file was provided
    data = spectro(channel(data, which='left'), fs, wl=wl, dB=NULL, plot=F,...)
  
  nsamples = length(data$time)*wl
  duration = nsamples/fs
  spectrum = data$amp
  
  min_freq1k = fmin/1000 
  max_freq1k = fmax/1000 
  
  which_min_freq = which(abs(data$freq - min_freq1k)==min(abs(data$freq - min_freq1k)))
  which_max_freq = which(abs(data$freq - max_freq1k)==min(abs(data$freq - max_freq1k))) 
  
  spectrum = spectrum[which_min_freq:which_max_freq,] # subset spectrum to [min,max]

  spectrum_rows = dim(spectrum)[1]
  spectrum_cols = dim(spectrum)[2]
  delta_tk   = (nsamples/fs) / spectrum_cols
  
  no_j    = floor(duration / j)
  I_per_j = floor(j/delta_tk) # number of values, in each row, for each j period (no. of columns)
  
  aci_fl   = rep(NA, spectrum_rows) # NOTE: changed from rep(NA, no_j) in
  # soundecology, which results in NA values if spectrum_rows < no_j 
  aci_vals = rep(NA, no_j)

  for (q_index in 1:spectrum_rows) { # for each frequency bin fl
    for (j_index in 1:no_j) { # for each j period of time
      min_col = j_index * I_per_j - I_per_j + 1
      max_col = j_index * I_per_j

      D = 0 # difference of values
      for (k in min_col:(max_col - 1))
        D = D + abs(spectrum[q_index,k] - spectrum[q_index,k + 1])

      sum_I = sum(spectrum[q_index, min_col:max_col])
      aci_vals[j_index] = D / sum_I
    }
    aci_fl[q_index] = sum(aci_vals)
  } 
  
  aci = sum(aci_fl)
  return(aci/(duration/60)) # divide by duration in minutes
}

################################################################################
# Acoustic Diversity and Evenness Indices (ADI/AEI)
# Only compatible with single-channel (mono) data
# Returns a list (ADI, AEI)
# Output identical to `acoustic_diversity` and `acoustic_evenness` from soundecology package
acidx_adei = function(
    data,           # A tuneR Wave object
    fs = 0,         # Sampling rate (hz)
    # fmin = 2000,  # TODO: Minimum frequency (hz)
    fmax = 8000,    # Maximum frequency (hz)
    fstep = 1000,   # Size of frequency bands (hz)
    threshold = -50 # Threshold (dBFS)
) {
  if (class(data)!='Wave') stop('tuneR Wave required')

  # Compute STFT with 10 hz per row
  hz_per_row = 10
  wl = fs/hz_per_row # TODO: check this
  data = spectro(channel(data, which='left'), f=fs, wl=wl, plot=F)
  
  # Proportion of values over the threshold in each band (hz)
  fbands = seq(from = 0, to = fmax - fstep, by = fstep)
  scores = rep(NA, length(fbands))
  for (i in 1:length(fbands)) {
    fbandmin  = round((fbands[i])/hz_per_row)
    fbandmax  = round((fbands[i] + fstep)/hz_per_row)
    fband     = data$amp[fbandmin:fbandmax,]
    scores[i] = length(fband[fband>threshold]) / length(fband)
  }

  adi = round(diversity(scores, index = 'shannon'), 6)
  aei = round(Gini(scores), 6)
  
  return(list(adi=adi, aei=aei))
}
