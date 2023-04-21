## Generate a csv mapping raw audio data files to metadata and inspect it
source('global.R')

## Generate data ------------------------------------------------------------
# In:  The DNR database drives
# Out: output/data.csv
library(stringr)
library(lubridate)

data = data.frame()
for (database_path in database_paths) {
  message('Scanning ', database_path)
  files = list.files(path=paste0(database_path), pattern="*.wav", full.names=T, recursive=T)
  
  # NOTE: extracting durations via tuneR too time intensive
  # durations = c()
  # i = 1
  # for (file in files) {
  #   message('(', i, ' of ', length(files), ') Reading ',  basename(file))
  #   wav = readWave(file)
  #   duration = round(length(wav@left) / wav@samp.rate) # in seconds
  #   durations = append(durations, duration)
  #   i = i + 1
  # }
  
  substrings = str_split(str_sub(basename(files), start = 1, end = -5), '_')
  serial = sapply(substrings, '[[', 1)
  unit = substring(serial, 1, 3)
  deployment = str_locate_all(pattern='Deployment', files)
  deployment = substring(files, sapply(deployment, '[[', 2)+1, sapply(deployment, '[[', 2)+1)
  
  date_raw = sapply(substrings, '[[', 2)
  year = substring(date_raw, 1, 4)
  date = ymd(date_raw) #as.POSIXct(date_raw, tz = tz, format='%Y%m%d')
  time = as.POSIXct(
    paste(date_raw, sapply(substrings, '[[', 3)),
    tz = tz, format='%Y%m%d %H%M%S')
  hour = format(round(time, units='hours'), format='%H')
  
  temp = data.frame(
    SerialNo   = serial,
    UnitType   = unit,
    DeployNo   = deployment,
    DataYear   = year,
    SurveyDate = date,
    DataTime   = time,
    NearHour   = hour,
    File       = files
  )
  data = rbind(data, temp)
}
if (nrow(data) != 75658) {
  stop('data length is unexpected!')
}
write.csv(data, file=recording_date_serial_output_path, row.names = FALSE)
message('Created ', recording_date_serial_output_path)

# Explore
# unique(data$SerialNo)
# unique(data$UnitType)
# unique(data$DeployNo)
# tapply(data$SurveyDate, data$SerialNo, unique)

## Show calendars of recorded dates --------------------------------------------
# library(calendR)
# for (year in unique(data$DataYear)) {
#   message('Creating calendar for ', year)
#   data_year = data[data$DataYear==year,]
#   print(calendR(year = year,
#           start_date = as.Date(data_year$SurveyDate)[1],
#           end_date   = as.Date(data_year$SurveyDate[nrow(data_year)]),
#           start = 'M'))
# }
