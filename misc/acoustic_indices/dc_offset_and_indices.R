library(soundecology)
library(tuneR)
library(seewave)
source('helpers.R')

# Load test data
path = '~/Desktop/oesf-examples/short'
files = list.files(path=path, pattern='*.wav', full.names=T, recursive=T)
file = files[2]

sound = readWave(file)
min(20*log10(abs(sound@left)/32768))
spectrogram(sound, interpolate = F)

filtered = ffilter(sound, from=80, output='Wave')
min(20*log10(abs(filtered@left)/32768))
spectrogram(filtered, interpolate = F)

dc_corrected = sound # if (dc_offset(sound@left))
dc_corrected@left = dc_correct(dc_corrected@left)
min(20*log10(abs(dc_corrected@left)/32768))
spectrogram(dc_corrected, interpolate = F)

# oscillogram(sound)
# oscillogram(filtered)
# oscillogram(dc_corrected)

# Parameters
min_freq = 2000 # Hz
max_freq = 10000
window   = 512  # FFT window size

# NOTE: Bioacoustic Index is approximately equal for raw sound and dc_corrected as long as
# min_freq is above the low-frequency distortion caused by a constant or slowly-changing DC offset
# i.e. dc correction has a negligible effect
for (s in list(sound, filtered, dc_corrected)) {
  # Calculate index
  bio = bioacoustic_index(
    soundfile = s,
    min_freq  = min_freq, # default 2000
    max_freq  = max_freq, # default 8000
    fft_w     = window
  )
  print(bio$left_area)
}

j = 5 # cluster size (sec)

# NOTE: ACI is approximately equal for raw sound and dc_corrected as long as
# min_freq is above the low-frequency distortion caused by a constant or slowly-changing DC offset
# i.e. dc correction has a negligible effect
for (s in list(sound, filtered, dc_corrected)) {
  # Calculate index
  aci_soundecology = acoustic_complexity(
    soundfile = s,
    min_freq  = min_freq,
    max_freq  = max_freq,
    fft_w     = window,
    j         = j
  )
  print(aci_soundecology$AciTotAll_left)
}

freq_step    = 100
db_threshold = -50
# NOTE: ADI and AEI is NOT equal between all three
for (s in list(sound, filtered, dc_corrected)) {
  adi = acoustic_diversity(
    soundfile = s,
    max_freq  = max_freq,
    freq_step = freq_step,
    db_threshold = db_threshold,
    shannon = TRUE
  )
  print(adi$adi_left)
}
for (s in list(sound, filtered, dc_corrected)) {
  aei = acoustic_evenness(
    soundfile = s,
    max_freq  = max_freq,
    freq_step = freq_step,
    db_threshold = db_threshold
  )
  print(aei$aei_left)
}

# Parameters
f1_min = 0
f1_max = 2000
f2_min = 2000
f2_max = 8000
# NOTE: NDSI is NOT equal between all three
for (s in list(sound, filtered, dc_corrected)) {
  # Calculate index
  ndsi_soundecology = ndsi(
    soundfile = s,
    fft_w     = window,
    anthro_min = f1_min,
    anthro_max = f1_max,
    bio_min    = f2_min,
    bio_max    = f2_max
  )
  print(ndsi_soundecology$ndsi_left)
}

# NOTE: H is NOT equal between all three
for (s in list(sound, filtered, dc_corrected)) {
  h = H(
    wave = s,
    wl = window
  )
  print(h)
}

# TODO: test effect of normalization / gain amplification on all indices