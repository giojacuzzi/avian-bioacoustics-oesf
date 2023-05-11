# Tests for optimized acoustic index functions

source('global.R')
source('acoustic_indices/acidx.R')
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
# benchmark_bio = microbenchmark(
#   'reference' = {
#     bioacoustic_index(soundfile, min_freq, max_freq, fft_w)
#   },
#   'custom (file)' = {
#     custom_bio_file = acidx_bio(soundfile, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq)
#   },
#   'custom (spectrogram)' = {
#     custom_bio_spec = acidx_bio(spectrogram, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq)
#   },
#   times = 100
# )
# summary(benchmark_bio)

# Eval ACI
reference_aci = acoustic_complexity(soundfile, min_freq, max_freq, j, fft_w)
nminutes = length(soundfile)/soundfile@samp.rate/60
reference_aci = reference_aci$AciTotAll_left/nminutes # divide by duration in minutes to compare
message('Reference: ', reference_aci)
spectrogram = spectro(soundfile, f=soundfile@samp.rate, wl=fft_w, plot=F, dB=NULL, wn='hamming')
custom_aci_spec = acidx_aci(spectrogram, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq, j=j)
message('Custom with spec: ', custom_aci_spec)
message(custom_aci_spec == reference_aci)
custom_aci_file = acidx_aci(soundfile, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq, j=j, wn='hamming')
message('Custom with file: ', custom_aci_file)
message(custom_aci_file == reference_aci)
# benchmark_aci = microbenchmark(
#   'reference' = {
#     acoustic_complexity(soundfile, min_freq, max_freq, j, fft_w)
#   },
#   'custom (file)' = {
#     custom_aci_file = acidx_aci(soundfile, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq, j=j, wn='hamming')
#   },
#   'custom (spectrogram)' = {
#     custom_aci_spec = acidx_aci(spectrogram, fs = soundfile@samp.rate, fmin = min_freq, fmax = max_freq, j=j)
#   },
#   times = 100
# )
# summary(benchmark_aci)


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
# benchmark_adei = microbenchmark(
#   'reference' = {
#     acoustic_diversity(soundfile, max_freq = max_freq, db_threshold = db_threshold, freq_step = freq_step)
#     acoustic_evenness(soundfile, max_freq = max_freq, db_threshold = db_threshold, freq_step = freq_step)
#   },
#   'custom' = {
#     custom_adei_file = acidx_adei(soundfile, fs = soundfile@samp.rate,fmax = max_freq,fstep = freq_step,threshold = db_threshold)
#   },
#   times = 100
# )
# summary(benchmark_adei)
