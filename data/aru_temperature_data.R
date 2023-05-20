## Extract temperature data from ARU summary txt files
source('global.R')

## Generate data ------------------------------------------------------------
# In:  The DNR database drives
# Out: output/aru_temperature_data.csv
library(stringr)
library(lubridate)

results = data.frame()

data = data.frame()
for (database_path in database_paths) {
  message('Scanning ', database_path)
  files = list.files(path=paste0(database_path), pattern="*Summary.txt", full.names=T, recursive=T)
  
  for (file in files) {
    substrings = str_split(str_sub(basename(file), start = 1, end = -5), '_')
    serial = sapply(substrings, '[[', 1)
    
    message(serial)
    
    data = read.csv(file)
    data = data[,c('DATE', 'TIME', 'TEMP.C.')]
    data$DATE = ymd(data$DATE)
    for (date in as.character(unique(data$DATE))) {
      data_date = data[data$DATE==date,]
      tmax = max(data_date$TEMP.C.)
      tmin = min(data_date$TEMP.C.)
      tmean = round(mean(data_date$TEMP.C.),2)
      # message(date, ' ', tmax, ' ', tmean, ' ', tmin)
      
      results = rbind(results, data.frame(
        SerialNo = serial,
        SurveyDate = date,
        TempMax = tmax,
        TempMean = tmean,
        TempMin = tmin
      ))
    }
  }
}

output_path = 'data/output/aru_temperature_data.csv'
write.csv(results, file=output_path, row.names = F)
message('Created ', output_path)
