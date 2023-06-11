source('global.R')
source('acoustic_indices/batch_process_acidx_df.R')

library(suncalc)
library(lubridate)

# 4 and 4 is good for working on something else (~1.45 min)
ncores = 6
batch_size = 6

data = get_recording_date_serial_data() #get_joined_data()

# Only look at 2020 for now
data = data[data$DataYear==2021, ]
data = data[!is.na(data$File),]

output_path = normalizePath('/Users/giojacuzzi/repos/olympic-songbirds/acoustic_indices/output/24hr/')
if (file.exists(paste0(output_path, '/files_processed.csv'))) {
  files_processed = read.csv(paste0(output_path, '/files_processed.csv'))
} else {
  files_processed = data.frame()
}

# TODO: fix erroneous DataYear that don't agree with SurveyDate

files_to_process = data.frame()

n_files_already_processed = 0

message('Determining files to process...')

for (date in as.character(unique(data$SurveyDate))) {
  date_data = data[data$SurveyDate==date,]
  
  for (serialno in unique(date_data$SerialNo)) {
    date_serial_data = date_data[date_data$SerialNo==serialno,]
    
    # EX: Get file corresponding to 06:00:00 hour: date_serial_data$NearHour==6
    files = date_serial_data[, 'File']
    
    if (length(files) == 0) {
      warning('File(s) for serialno ', serialno, ' on date ', date, ' not found. Skipping.')
      next
    }
    
    if (sum(files %in% files_processed$File) != 0) {
      n_files_already_processed = n_files_already_processed + 1
      next
    }
    
    output_file = paste0(date, '_', serialno) # TODO: remove

    files_to_process = rbind(files_to_process, data.frame(
      File = files,
      SurveyDate = date,
      SerialNo = serialno
    ))
  }
}

message(nrow(files_to_process), ' files to process (', n_files_already_processed, ' files already processed)')

for (date in as.character(sort(unique(as.Date(files_to_process$SurveyDate))))) {
  # Do batch process for this date and save to file
  # N-minute intervals
  # FFT window size of 512
  # BIO:
  # min_freq = 1700, max_freq = 10000
  # ACI:
  # min_freq = 1700, max_freq = , j = 5
  
  message('Date ', date)
  
  
  date_data = files_to_process[files_to_process$SurveyDate==date, ]
  files = date_data$File
  
  if (length(files) == 0) {
    warning('File(s) for date ', date, ' not found. Skipping.')
    next
  }
  
  batches = split(files, ceiling(seq_along(files)/batch_size))
  for (batch in batches) {
    get_serial_from_file_name(batch)
    message(' SerialNos ', paste(get_serial_from_file_name(batch), collapse = ' '))
    
    alpha_indices = c('ACI', 'BIO') # 'NDSI'
    process_data = batch_process_acidx(
      batch,
      output_path = output_path, # TODO: remove
      output_file = output_file, # TODO: remove
      alpha_indices = alpha_indices,
      wl = 512,
      time_interval = 60 * 10,
      ncores = ncores,
      # ACI, BIO
      min_freq = 1700,
      max_freq = 10000,
      j = 5
      # # NDSI
      # anthro_min = 1,
      # anthro_max = 1700,
      # bio_min = 1700,
      # bio_max = 10000
    )
    
    results = process_data$results
    results$SerialNo = get_serial_from_file_name(results$File)
    results$SurveyDate = date
    results$TimeStart = get_time_from_file_name(results$File) + results$TimeStart
    results$TimeEnd = get_time_from_file_name(results$File) + results$TimeEnd
    output_results = paste0(output_path, '/results.csv')
    write.table(results,
                file = output_results, sep = ',',
                append = file.exists(output_results),
                col.names = !file.exists(output_results),
                row.names = F)
    
    diagnostics = process_data$diagnostics
    diagnostics$SerialNo = get_serial_from_file_name(results$File)
    diagnostics$SurveyDate = date
    output_diagnostics = paste0(output_path, '/diagnostics.csv')
    write.table(diagnostics,
                file = output_diagnostics, sep = ',',
                append = file.exists(output_diagnostics),
                col.names = !file.exists(output_diagnostics),
                row.names = F)
    
    files_processed = data.frame(File=process_data$files_processed)
    output_files_processed = paste0(output_path, '/files_processed.csv')
    write.table(files_processed,
                file = output_files_processed, sep = ',',
                append = file.exists(output_files_processed),
                col.names = !file.exists(output_files_processed),
                row.names = F)
    
    rm(process_data)
  }
}
