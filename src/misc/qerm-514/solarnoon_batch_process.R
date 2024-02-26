source('global.R')
source('acoustic_indices/batch_process_acidx_df.R')

library(suncalc)
library(lubridate)

data = get_joined_data()

output_path = '/Users/giojacuzzi/repos/olympic-songbirds/acoustic_indices/output/solar_noon/'
files_processed = read.csv(paste0(output_path, 'files_processed.csv'))

# TODO: fix erroneous DataYear that don't agree with SurveyDate

for (date in as.character(unique(data$SurveyDate))) {
  date_data = data[data$SurveyDate==date,]
  
  message('Date ', date)
  
  for (serialno in unique(date_data$SerialNo)) {
    date_serial_data = date_data[date_data$SerialNo==serialno,]
    
    # Get file corresponding to solarNoon hour:
    solarNoon = getSunlightTimes(date = date_data$SurveyDate[1], lat = 47.73533, lon = -124.24154,
                               tz = 'America/Los_Angeles')$solarNoon
    hr_during = floor_date(solarNoon, unit = 'hours')       # hour during solarNoon

    solarNoon_files = date_serial_data[
      round_date(date_serial_data$DataTime, 'hour')
      %in% c(hr_during), 'File']
    
    if (length(solarNoon_files) == 0) {
      warning('File(s) for date ', solarNoon, ' not found. Skipping.')
      next
    }
    
    if (solarNoon_files %in% files_processed$File) {
      message('Files(s) for date ', solarNoon, ' already processed. Skipping.')
      next
    }
    
    output_file = paste0(date, '_', serialno)
    message(' SerialNo ', serialno)
    
    # Do batch process for this date and serialno and save to file
    # N-minute intervals
    # FFT window size of 512
    # BIO:
    # min_freq = 1700, max_freq = 10000
    # ACI:
    # min_freq = 1700, max_freq = , j = 5

    alpha_indices = c('ACI', 'BIO')
    process_data = batch_process_acidx(
      solarNoon_files,
      output_path = output_path,
      output_file = output_file,
      alpha_indices = alpha_indices,
      wl = 512,
      time_interval = 60 * 60,
      ncores = 1,
      min_freq = 1700,
      max_freq = 10000,
      j = 5
    )
    
    results = process_data$results
    results$SerialNo = serialno
    results$SurveyDate = date
    results$TimeStart = get_time_from_file_name(results$File) + results$TimeStart
    results$TimeEnd = get_time_from_file_name(results$File) + results$TimeEnd
    output_results = paste0(output_path, 'results.csv')
    write.table(results,
                file = output_results, sep = ',',
                append = file.exists(output_results),
                col.names = !file.exists(output_results),
                row.names = F)

    diagnostics = process_data$diagnostics
    diagnostics$SerialNo = serialno
    diagnostics$SurveyDate = date
    output_diagnostics = paste0(output_path, 'diagnostics.csv')
    write.table(diagnostics,
                file = output_diagnostics, sep = ',',
                append = file.exists(output_diagnostics),
                col.names = !file.exists(output_diagnostics),
                row.names = F)

    files_processed = data.frame(File=process_data$files_processed)
    output_files_processed = paste0(output_path, 'files_processed.csv')
    write.table(files_processed,
                file = output_files_processed, sep = ',',
                append = file.exists(output_files_processed),
                col.names = !file.exists(output_files_processed),
                row.names = F)
  }
}
