# Extract temperature data from Wildlife Acoustics summary files
source('global.R')

files = c(
  list.files('/Volumes/GIOJ Backup/DNR', pattern = '*.txt', recursive=T, full.names=T),
  list.files('/Volumes/SAFS Backup/DNR', pattern = '*.txt', recursive=T, full.names=T)
)

results = data.frame()
i = 1
while (i <= length(files)) {
  file = files[i]
  
  message('(', i, ' of ', length(files), ') Reading ', file)
  i = i+1
  
  SerialNo = get_serial_from_file_name(file)
  UnitType = substring(SerialNo, 1, 3)

  cols = c('DATE', 'TIME', 'TEMP_C')
  
  data = tryCatch({
    
    if (UnitType=='SM2') {
      data = na.omit(read.table(file, fill=T)[1:3])
      colnames(data) = cols
      data
    } else {
      data = read.table(file, skip = 1, sep = ',', col.names =
                   c('DATE','TIME','LAT','LATC','LON','LONC','POWER_V',
                     'TEMP_C','NUM_FILES','MIC0 TYPE','MIC1 TYPE'))
      data = data[,cols]
    }
  }, error = function(e) {
    message(paste('Error: ', e))
    NULL
  })
  if (is.null(data)) {
    stop()
  }
  
  malformatted_rows = which(is.na(as.Date(data$DATE, '%Y-%b-%d')))
  if (length(malformatted_rows) > 0) {
    data = data[-malformatted_rows, ]
  }
  data = na.omit(data)

  SurveyDate = as.Date(data$DATE, '%Y-%b-%d')
  DataTime = as.POSIXct(paste(SurveyDate, data$TIME), tz=tz)

  results = rbind(results, data.frame(
    SerialNo,
    SurveyDate,
    DataTime,
    Temp = data$TEMP_C
  ))
}

message('Finished! Result stats in degrees Fahrenheit')
print(summary(results$Temp * 9/5 + 32))

write.csv(results, 'data/output/recording_date_temperature_data.csv')
