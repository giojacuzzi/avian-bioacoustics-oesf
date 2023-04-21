source('acoustic_indices/batch_process_alpha_indices.R')

path = '~/../../Volumes/SAFS Work/DNR/test/MATURE'

interval = 60 * 2

# NOTE: runtime ~ 1 hr
batch_process_alpha_indices(
  input_files = list.files(path=path, pattern='*.wav', full.names=T, recursive=F),
  output_path = paste0(path,'/output/'),
  output_file = 'MATURE',
  alpha_indices = c('ACI','ADI','AEI','BIO','H','M','NDSI'),
  time_interval = interval,
  ncores = 12,
  dc_correct = T,
  digits = 4,
  diagnostics = T,
  # Alpha index parameters
  min_freq = 2000,
  max_freq = 8000,
  anthro_min = 100,
  anthro_max = 2000, 
  bio_min = 2000,
  bio_max = 8000,
  db_threshold = -40, 
  freq_step = 1000,
  j = 5,
  fft_w = 512,
  wl = 512
)

source('helpers.R')
diagnostics = read.csv('~/../../Volumes/SAFS Work/DNR/test/MATURE/output/MATURE_diagnostics.csv')
summary(linear_to_dBFS(diagnostics$DcBias))

source('global.R')
data = read.csv('~/../../Volumes/SAFS Work/DNR/test/MATURE/output/MATURE.csv')
data$ACI = data$ACI/interval # normalize by interval
data$Time = as.POSIXct(paste(get_date_from_file_name(data$File), get_hour_from_file_name(data$File)),
                       tz=tz, format='%Y-%m-%d %H') + data$Start

library(tidyr)
data_long = gather(data, 'index', 'value', c('BIO','ADI','AEI', 'M','H','ACI','NDSI'))
data_long$index = factor(data_long$index)

library(ggplot2)
theme_set(theme_minimal())
ggplot(data, aes(x=Time, y=ACI)) + geom_point() + geom_smooth() + labs(title='ACI')
ggplot(data, aes(x=Time, y=ADI)) + geom_point() + geom_smooth() + labs(title='ADI')
ggplot(data, aes(x=Time, y=AEI)) + geom_point() + geom_smooth() + labs(title='AEI')
ggplot(data, aes(x=Time, y=BIO)) + geom_point() + geom_smooth() + labs(title='BIO')
ggplot(data, aes(x=Time, y=M)) + geom_point() + geom_smooth() + labs(title='M')
ggplot(data, aes(x=Time, y=H)) + geom_point() + geom_smooth() + labs(title='H')
ggplot(data, aes(x=Time, y=NDSI)) + geom_point() + geom_smooth() + labs(title='NDSI')
