source('acoustic_indices/batch_process_alpha_indices.R')
source('global.R')

site_data = get_joined_data()

# Get all units active on a particular SurveyDate in a particular Strata
SurveyDate = '2020-05-28'

Strata     = 'STAND INIT' # COMP EXCL, MATURE, STAND INIT, THINNED
active_serialnos = unique(site_data[site_data$SurveyDate==SurveyDate & site_data$Strata==Strata,'SerialNo'])
# Get the audio files for the first unit, in a particular time range
unit = active_serialnos[1]
HourInterval = c(3,8)
files = site_data[site_data$SurveyDate==SurveyDate &
                         site_data$Strata==Strata &
                         (site_data$NearHour>= HourInterval[1] &
                            site_data$NearHour<= HourInterval[2]) &
                         site_data$SerialNo==unit, 'File']
# Optionally, view the raw files yourself:
# utils::browseURL(dirname(files)[1])

indices = c('ACI','BIO','ADI', 'AEI', 'H')
interval = 60 * 2 # sec

outpath = '~/Desktop/oesf-examples/output/'
outfile = paste(Strata, unit)

# Parameters
thresh_db = -45
min_f = 2000
max_f = 9000
step_f = 1000
window = 512

# View spectrogram of an hour
# spectrogram(readWave(files[4]), alim = c(db_threshold, 0))

batch_process_alpha_indices(
  input_files   = files,
  output_path   = outpath,
  output_file   = outfile,
  alpha_indices = indices,
  time_interval = interval,
  ncores        = 12,
  dc_correct    = T,
  digits        = 4,
  diagnostics   = T,
  # Alpha index parameters
  min_freq = min_f,
  max_freq = max_f,
  anthro_min = 10,
  anthro_max = min_f, 
  bio_min = min_f,
  bio_max = max_f,
  db_threshold = thresh_db, 
  freq_step = step_f,
  j = 5,
  fft_w = window,
  wl = window
)

source('helpers.R')
diagnostics = read.csv(paste0(outpath,outfile,'_diagnostics.csv'))
summary(linear_to_dBFS(diagnostics$DcBias))

data = read.csv(paste0(outpath,outfile,'.csv'))
data$ACI = data$ACI / (interval/60) # normalize by interval length in minutes
data$Time = as.POSIXct(paste(get_date_from_file_name(data$File), get_hour_from_file_name(data$File)),
                       tz=tz, format='%Y-%m-%d %H') + data$Start

library(ggplot2)
theme_set(theme_minimal())
ggplot(data, aes(x=Time, y=ACI)) + geom_line() + geom_smooth() + labs(title = 'ACI', subtitle = paste(SurveyDate, Strata, unit))
ggplot(data, aes(x=Time, y=BIO)) + geom_line() + geom_smooth() + labs(title = 'BIO', subtitle = paste(SurveyDate, Strata, unit))
ggplot(data, aes(x=Time, y=ADI)) + geom_line() + geom_smooth() + labs(title = 'ADI', subtitle = paste(SurveyDate, Strata, unit))
ggplot(data, aes(x=Time, y=AEI)) + geom_line() + geom_smooth() + labs(title = 'AEI', subtitle = paste(SurveyDate, Strata, unit))
ggplot(data, aes(x=Time, y=H)) + geom_line() + geom_smooth() + labs(title = 'H', subtitle = paste(SurveyDate, Strata, unit))
ggplot(data, aes(x=Time, y=NDSI)) + geom_line() + geom_smooth() + labs(title = 'NDSI', subtitle = paste(SurveyDate, Strata, unit))




###-----------------
# path = '~/../../Volumes/SAFS Work/DNR/test/MATURE'
# path = '~/Desktop/oesf-examples'
# path = '~/../../Volumes/SAFS Work/DNR/test/MATURE'
# outfile = 'test'
# 
# indices = c('ACI','ADI','AEI','BIO','H')
# interval = 60 * 2
# 
# # NOTE: runtime ~ 1 hr
# batch_process_alpha_indices(
#   input_files = list.files(path=path, pattern='*.wav', full.names=T, recursive=F),
#   output_path = paste0(path,'/output/'),
#   output_file = outfile,
#   alpha_indices = indices,
#   time_interval = interval,
#   ncores = 12,
#   dc_correct = T,
#   digits = 4,
#   diagnostics = T,
#   # Alpha index parameters
#   min_freq = 2000,
#   max_freq = 8000,
#   anthro_min = 100,
#   anthro_max = 2000, 
#   bio_min = 2000,
#   bio_max = 8000,
#   db_threshold = -40, 
#   freq_step = 1000,
#   j = 5,
#   fft_w = 512,
#   wl = 512
# )
# 
# source('helpers.R')
# diagnostics = read.csv(paste0(path,'/output/',outfile,'_diagnostics.csv'))
# summary(linear_to_dBFS(diagnostics$DcBias))
# 
# source('global.R')
# data = read.csv(paste0(path,'/output/',outfile,'.csv'))
# data$ACI = data$ACI/interval # normalize by interval
# data$Time = as.POSIXct(paste(get_date_from_file_name(data$File), get_hour_from_file_name(data$File)),
#                        tz=tz, format='%Y-%m-%d %H') + data$Start
# 
# library(tidyr)
# data_long = gather(data, 'index', 'value', indices)
# data_long$index = factor(data_long$index)
# 
# library(ggplot2)
# theme_set(theme_minimal())
# ggplot(data, aes(x=Time, y=ACI)) + geom_point() + geom_smooth() + labs(title='ACI')
# ggplot(data, aes(x=Time, y=ADI)) + geom_point() + geom_smooth() + labs(title='ADI')
# ggplot(data, aes(x=Time, y=AEI)) + geom_point() + geom_smooth() + labs(title='AEI')
# ggplot(data, aes(x=Time, y=BIO)) + geom_point() + geom_smooth() + labs(title='BIO')
# ggplot(data, aes(x=Time, y=M)) + geom_point() + geom_smooth() + labs(title='M')
# ggplot(data, aes(x=Time, y=H)) + geom_point() + geom_smooth() + labs(title='H')
# ggplot(data, aes(x=Time, y=NDSI)) + geom_point() + geom_smooth() + labs(title='NDSI')
