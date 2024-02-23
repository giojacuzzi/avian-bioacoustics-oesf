source('global.R')
source('acoustic_indices/batch_process_acidx.R')

library(suncalc)
library(lubridate)

data = get_joined_data()

output_path = '/Users/giojacuzzi/repos/olympic-songbirds/acoustic_indices/output/'

# Only choose dates from 2021 for now, and only do the hour before, during, and after sunrise

# TODO: fix erroneous DataYear that don't agree with SurveyDate

data = data[format(data$SurveyDate, '%Y')==2021,]

for (date in as.character(unique(data$SurveyDate))) {
  date_data = data[data$SurveyDate==date,]
  
  message('Date ', date)
  
  for (serialno in unique(date_data$SerialNo)) {
    date_serial_files = date_data[date_data$SerialNo==serialno,'File']
    
    # Get files corresponding to sunrise hours:
    sunrise = getSunlightTimes(date = date, lat = 47.73533, lon = -124.24154,
                               tz = 'America/Los_Angeles')$sunrise
    hr_before = floor_date(sunrise-60*60, unit = 'hours') # hour before sunrise
    hr_during = floor_date(sunrise, unit = 'hours')       # hour during sunrise
    hr_after  = floor_date(sunrise+60*60, unit = 'hours') # hour after sunrise
    
    sunrise_files = date_serial_files[
      round_date(date_serial_files$DataTime, 'hour')
      %in% c(hr_before, hr_during, hr_after), 'File']
    
    
    output_file = paste0(date, '_', serialno)
    message(' SerialNo ', serialno)
    
    # Do batch process for this date and serialno and save to file
    # 10-minute intervals
    # FFT window size of 512
    # BIO:
    # min_freq = 1700, max_freq = 10000
    # ACI:
    # min_freq = 1700, max_freq = , j = 5
    
    # Only do ACI for now
    alpha_indices = c('ACI')
    batch_process_acidx(
      sunrise_files,
      output_path = output_path,
      output_file = output_file,
      alpha_indices = alpha_indices,
      wl = 512,
      time_interval = 60*10, # 10-minute time intervals
      ncores = 14,
      min_freq = 1700,
      max_freq = 10000,
      j = 5
    )
  }
}

