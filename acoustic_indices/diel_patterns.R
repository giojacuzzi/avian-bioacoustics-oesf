source('acoustic_indices/batch_process_alpha_indices.R')
source('global.R')

site_data = get_joined_data()

outpath = '~/../../Volumes/SAFS Work/DNR/2020/Deployment4_May20_24/_output/'

# Get all units active on a particular SurveyDate in a particular Strata
SurveyDate = '2020-05-28'

# Option: active_serialnos = unique(site_data[site_data$SurveyDate==SurveyDate,c('SerialNo','Strata')])
# Option: Get the first serial number of each strata (~40 minutes for 24 files)
# first_per_strata = sapply(tapply(active_serialnos$SerialNo, active_serialnos$Strata, function(x) { x }), '[[', 1)

# Take the files from the full database drives
# files = site_data[site_data$SurveyDate==SurveyDate &
#                     (site_data$NearHour>= HourInterval[1] &
#                        site_data$NearHour<= HourInterval[2]) &
#                     site_data$SerialNo %in% active_serialnos$SerialNo, 'File']

# Option: Alternatively, take files from SAFS Work drive
files = list.files(path='~/../../Volumes/SAFS Work/DNR/2020/Deployment4_May20_24', pattern="*.wav", full.names=T, recursive=T)
to_process = data.frame(
  File=files,
  SerialNo=site_data[match(basename(files), basename(site_data$File)), 'SerialNo'],
  Strata=site_data[match(basename(files), basename(site_data$File)), 'Strata'],
  SurveyDate=site_data[match(basename(files), basename(site_data$File)), 'SurveyDate'],
  NearHour=site_data[match(basename(files), basename(site_data$File)), 'NearHour'],
  DataTime=site_data[match(basename(files), basename(site_data$File)), 'DataTime']
)

# Only process specific hours
HourInterval = c(3,5) # inclusive
to_process = to_process[(to_process$NearHour >= HourInterval[1] &
                           to_process$NearHour <= HourInterval[2]),]

# Alpha acoustic index arameters
indices = c('ACI','BIO','ADI', 'AEI', 'H', 'M', 'NDSI')
interval = 60 * 5 # in sec (i.e. seconds * minutes)
thresh_db = -45
min_f = 2000
max_f = 9000
step_f = 1000
window = 512

# View spectrogram of an hour
# spectrogram(readWave(files[4]), alim = c(db_threshold, 0))

########## PROCESS ##########
i = 1
for (serial in unique(to_process$SerialNo)) {
  
  outfile = paste0(serial,'_',SurveyDate) #'diel_patterns' #SurveyDate #paste(Strata, unit)
  infiles = to_process[to_process$SerialNo==serial,'File']
  
  message('Processing ', serial, ' (', i, ' of ', length(unique(to_process$SerialNo)),' sites)...')

  batch_process_alpha_indices(
    input_files   = infiles,
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
  i = i+1
}
##############################

# Fetch results
results_files    = list.files(path=outpath, pattern="*.csv", full.names=T, recursive=F)
diagnostic_files = results_files[grepl('diagnostics', results_files, fixed = T)]
results_files    = results_files[!grepl('diagnostics', results_files, fixed = T)]

source('helpers.R')
library(ggplot2)
theme_set(theme_minimal())

# Explore diagnostics
diagnostics = data.frame()
for (d in diagnostic_files) {
  diagnostics = rbind(diagnostics, read.csv(d, header=T))
}
summary(diagnostics$Clipping)
summary(linear_to_dBFS(diagnostics$DcBias))
ggplot(diagnostics, aes(x=linear_to_dBFS(DcBias))) + geom_histogram()

# Explore data
data = data.frame()
for (r in results_files) {
  data = rbind(data, read.csv(r, header=T))
}
data = full_join(data, to_process, by = c('File'))
data$DataTime = as.POSIXct(data$DataTime, tz=tz) + data$Start # calculate actual start times of each interval
data$Strata = factor(data$Strata)
data$SerialNo = factor(data$SerialNo)
summary(data$Strata)

ggplot(data, aes(x=DataTime, y=ACI, color=Strata, linetype=SerialNo)) + geom_line()
ggplot(data, aes(x=DataTime, y=BIO, fill=Strata)) + geom_boxplot()

# TODO: ts? aggregate?

# Apply rolling mean to data to smooth
# library(zoo)
# k = interval / (10) # width of rolling window (interval/x sec)
# mean_data <- data %>%
#   group_by(Strata) %>% 
#   mutate(ACI_mean = rollmean(ACI, k, na.pad = T)) %>%
#   mutate(ADI_mean = rollmean(ADI, k, na.pad = T)) %>%
#   mutate(AEI_mean = rollmean(AEI, k, na.pad = T)) %>%
#   mutate(BIO_mean = rollmean(BIO, k, na.pad = T)) %>%
#   mutate(H_mean = rollmean(H, k, na.pad = T))
# ggplot(mean_data, aes(x=Time, y=ACI_mean, color=Strata)) + geom_line() + labs(title = 'ACI')
# ggplot(mean_data, aes(x=Time, y=ADI_mean, color=Strata)) + geom_line() + labs(title = 'ADI')
# ggplot(mean_data, aes(x=Time, y=AEI_mean, color=Strata)) + geom_line() + labs(title = 'AEI')
# ggplot(mean_data, aes(x=Time, y=BIO_mean, color=Strata)) + geom_line() + labs(title = 'BIO')
# ggplot(mean_data, aes(x=Time, y=H_mean, color=Strata)) + geom_line() + labs(title = 'H')
