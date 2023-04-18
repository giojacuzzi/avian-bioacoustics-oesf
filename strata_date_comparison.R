source('global.R')

date = '2021-05-02'

data = get_joined_data()
date_data = data[data$SurveyDate==date,]

serial_nos = c('SMA00399', # STAND INIT
               'SMA00393', # COMP EXCL
               'SMA00310', # THINNED
               'SMA00351') # MATURE

date_data = date_data[date_data$SerialNo %in% serial_nos,]

files = date_data$File
path = paste0(database_paths[1], '/2021/')

library(soundecology)
library(seewave)
library(tuneR)

results = data.frame()
for (serial_no in serial_nos) { # For each site
  message('Site ', serial_no)
  
  date_site_data = date_data[date_data$SerialNo==serial_no,]
  files = list.files(path=database_paths[1], pattern=paste0(serial_no, '_20210502_.*.wav'), full.names=T, recursive=T)
  
  # For each hour at the site
  for (hr in 1:nrow(date_site_data)) {
    message(hr, '...')
    
    if (hr != 5) next # only look at a specific hour for now

    # Load file
    file = list.files(path=database_paths[1], pattern=paste0(date_site_data[hr,'File']), full.names=T, recursive=T)
    wav = readWave(file)
    wav@left = wav@left - mean(wav@left) # remove DC offset
    
    # Specify params
    min_freq   = 2000
    max_freq   = 8000
    freq_step  = 1000 # ADI/AEI
    anthro_min = 20   # NDSI
    anthro_max = min_freq
    bio_min    = anthro_max
    bio_max    = max_freq
    window     = 512
    
    # Calculate indices
    bio = bioacoustic_index(
      soundfile = wav,
      min_freq  = min_freq,
      max_freq  = max_freq,
      fft_w     = window
    )
    adi = acoustic_diversity(
      soundfile = wav,
      max_freq  = max_freq,
      freq_step = freq_step
    )
    aei = acoustic_evenness(
      soundfile = wav,
      max_freq  = max_freq,
      freq_step = freq_step
    )
    aci = acoustic_complexity(
      soundfile = wav,
      min_freq  = min_freq,
      max_freq  = max_freq,
      fft_w     = window
    )
    h = H(
      wave = wav,
      wl   = window
    )
    ndsi = ndsi(
      soundfile  = wav,
      fft_w      = window,
      anthro_min = anthro_min,
      anthro_max = anthro_max,
      bio_min    = bio_min,
      bio_max    = bio_max
    )

    # Store result
    results = rbind(results, data.frame(
      SerialNo = serial_no,
      Strata   = date_site_data[1,'Strata'],
      Hour     = date_site_data[1,'NearHour'],
      BIO      = bio$left_area,
      ADI      = adi$adi_left,
      AEI      = aei$aei_left,
      ACI      = aci$AciTotAll_left,
      H        = h,
      NDSI     = ndsi$ndsi_left
    ))
    message(bio$left_area)
  }
}