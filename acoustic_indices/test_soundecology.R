library(seewave) # https://cran.r-project.org/web/packages/seewave/index.html
library(soundecology) # https://cran.r-project.org/web/packages/soundecology/vignettes/intro.html

# https://cran.r-project.org/web/packages/soundecology/vignettes/ACIandSeewave.html
# https://www.spectraplus.com/DT_help/overlap_percentage.htm

#Call the Wave object into memory
data(tropicalsound)

# Bioacoustic Index
BI_soundecology = bioacoustic_index(tropicalsound)
print(BI_soundecology$left_area)
summary(BI_soundecology)

# Acoustic Diversity Index
ADI_soundecology = acoustic_diversity(tropicalsound)
print(ADI_soundecology$adi_left)
summary(ADI_soundecology)

# Acoustic Evenness Index
AEI_soundecology = acoustic_evenness(tropicalsound)
print(AEI_soundecology$aei_left)
summary(AEI_soundecology)

# Acoustic Complexity Index
ACI_soundecology = acoustic_complexity(
  soundfile = tropicalsound, # Wave object
  min_freq = NA, # min frequency (Hz)
  max_freq = NA, # max frequency (Hz)
  j = 5,         # cluster size (sec)
  fft_w = 512    # window length
)
print(ACI_soundecology$AciTotAll_left)
summary(ACI_soundecology)

# Acoustic Complexity Index
ACI_seewave = ACI(
  wave = tropicalsound, # an R object
  f = -1,         # sampling frequency (Hz)
  channel = 1,    # channel (1==right)
  wl = 512,       # window length
  ovlp = 0,       # overlap between two successive windows (%)
  wn = 'hamming', # window type
  flim = NULL,    # frequency band min/max
  nbwindows = 1   # num windows
)

# Normalized Difference Soundscape Index
NDSI_soundecology = ndsi(
  soundfile = tropicalsound, # Wave object
  fft_w = 1024,              # fft window size
  anthro_min = 1000,         # min range of anthrophony (Hz)
  anthro_max = 2000,         # max
  bio_min = 2000,            # min range of biophony (Hz)
  bio_max = 11000            #
)
print(NDSI_soundecology$ndsi_left)
summary(NDSI_soundecology)

NDSI_seewave = NDSI(
  x = , # matrix computed with soundscapespec
  anthropophony = 1, # frequency bands of anthrophony (kHz)
  biophony = 2:8, # frequency bands of biophony (kHz)
  max = FALSE # if true, defines biophony as max, not sum, of 2 and 8 kHz bands
)
