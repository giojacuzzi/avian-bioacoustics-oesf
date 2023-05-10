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
library(profvis)
library(vegan)
library(ineq)
library(microbenchmark)

soundfile = readWave('~/Desktop/oesf-examples/short/04_short.wav')
################################################################################
min_freq = 1700
max_freq = 10000
fft_w = 512
j = 5

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

# Eval BIO
reference_bio = bioacoustic_index(soundfile, min_freq, max_freq, fft_w)
message('Reference: ', reference_bio$left_area)
spectrogram = spectro(soundfile, f=soundfile@samp.rate, wl=fft_w, plot=F, dB=NULL)
custom_bio_spec = acidx_bio(spectrogram, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq)
message('Custom with spec: ', custom_bio_spec)
message(custom_bio_spec == reference_bio$left_area)
custom_bio_file = acidx_bio(soundfile, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq)
message('Custom with file: ', custom_bio_file)
message(custom_bio_file == reference_bio$left_area)
benchmark_bio = microbenchmark(
  'reference' = {
    bioacoustic_index(soundfile, min_freq, max_freq, fft_w)
  },
  'custom (file)' = {
    custom_bio_file = acidx_bio(soundfile, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq)
  },
  'custom (spectrogram)' = {
    custom_bio_spec = acidx_bio(spectrogram, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq)
  },
  times = 100
)
summary(benchmark_bio)

################################################################################
# Acoustic Complexity Index (ACI)
# Only compatible with single-channel (mono) data
# Output identical to `acoustic_complexity` from soundecology package when wn='hamming' specified
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
    data = spectro(channel(soundfile, which='left'), fs, wl=wl, dB=NULL, plot=F,...)
  
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
  
  aci_vals = rep(NA, no_j)
  aci_fl   = rep(NA, no_j)
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
  return(aci) # TODO: return value / duration in minutes (/ duration * 60)
}

# Eval ACI
reference_aci = acoustic_complexity(soundfile, min_freq, max_freq, j, fft_w)
message('Reference: ', reference_aci$AciTotAll_left)
spectrogram = spectro(soundfile, f=soundfile@samp.rate, wl=fft_w, plot=F, dB=NULL, wn='hamming')
custom_aci_spec = acidx_aci(spectrogram, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq, j=j)
message('Custom with spec: ', custom_aci_spec)
message(custom_aci_spec == reference_aci$AciTotAll_left)
custom_aci_file = acidx_aci(soundfile, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq, j=j, wn='hamming')
message('Custom with file: ', custom_aci_file)
message(custom_aci_file == reference_aci$AciTotAll_left)
benchmark_aci = microbenchmark(
  'reference' = {
    acoustic_complexity(soundfile, min_freq, max_freq, j, fft_w)
  },
  'custom (file)' = {
    custom_aci_file = acidx_aci(soundfile, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq, j=j, wn='hamming')
  },
  'custom (spectrogram)' = {
    custom_aci_spec = acidx_aci(spectrogram, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq, j=j)
  },
  times = 100
)
summary(benchmark_aci)

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

# Eval ADI and AEI
db_threshold = -50
freq_step = 1000
reference_adi = acoustic_diversity(soundfile, max_freq = max_freq, db_threshold = db_threshold, freq_step = freq_step)
reference_aei = acoustic_evenness(soundfile, max_freq = max_freq, db_threshold = db_threshold, freq_step = freq_step)
message('Reference ADI: ', reference_adi$adi_left)
message('Reference AEI: ', reference_aei$aei_left)
custom_adei_file = acidx_adei(soundfile, fs = soundfile@samp.rate,fmax = max_freq,fstep = freq_step,threshold = db_threshold)
message('Custom with file: ', custom_adei_file$adi, ', ', custom_adei_file$aei)
message(custom_adei_file$adi == reference_adi$adi_left)
message(custom_adei_file$aei == reference_aei$aei_left)
benchmark_adei = microbenchmark(
  'reference' = {
    acoustic_diversity(soundfile, max_freq = max_freq, db_threshold = db_threshold, freq_step = freq_step)
    acoustic_evenness(soundfile, max_freq = max_freq, db_threshold = db_threshold, freq_step = freq_step)
  },
  'custom' = {
    custom_adei_file = acidx_adei(soundfile, fs = soundfile@samp.rate,fmax = max_freq,fstep = freq_step,threshold = db_threshold)
  },
  times = 100
)
summary(benchmark_adei)
