# Extract temperature data from Wildlife Acoustics summary files
source('global.R')

files = list.files('/Volumes/SAFS Backup/DNR', pattern = "*.txt", recursive = T, full.names = T)
# TODO: include files from both drives

results = data.frame()
i = 1
for (file in files) {
  message('(', i, ' of ', length(files), ') Reading ', file)
  i = i+1
  
  data = tryCatch({
    read.table(file, skip = 1, sep = ',', col.names =
                 c('DATE','TIME','LAT','LATC','LON','LONC','POWER_V',
                   'TEMP_C','NUM_FILES','MIC0 TYPE','MIC1 TYPE'))
  }, error = function(e) {
    warning(paste('Warning: ', e))
    NULL
  })
  if (is.null(data)) {
    message('Error reading file, skipping...')
    next
  }

  SerialNo = get_serial_from_file_name(file)
  SurveyDate = as.Date(data$DATE, '%Y-%b-%d')
  DataTime = as.POSIXct(paste(SurveyDate, data$TIME), tz=tz)

  results = rbind(results, data.frame(
    SerialNo,
    SurveyDate,
    DataTime,
    Temp = data$TEMP_C
  ))
}

# Print some stats in degrees Fahrenheit
message('Finished!')
print(summary(results$Temp * 9/5 + 32))

# TODO: save results to .csv
