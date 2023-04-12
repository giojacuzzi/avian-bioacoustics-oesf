## Generate a csv mapping raw audio data files to metadata and inspect it

# Paths to directories comprising the DNR database
database_paths = c(
  '~/../../Volumes/GIOJ Backup/DNR',
  '~/../../Volumes/SAFS Backup/DNR'
)
database_path_working = '~/../../Volumes/SAFS Work/DNR'

## Generate filemap ------------------------------------------------------------
# In:  The DNR database drives
# Out: output/filemap.csv
library(stringr)

output_path = 'data/output/filemap.csv'
tz = 'America/Los_Angeles'

filemap = data.frame()
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
  date = as.POSIXct(date_raw, tz = tz, format='%Y%m%d')
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
    File = basename(files)
  )
  filemap = rbind(filemap, temp)
}
if (nrow(filemap) != 75658) {
  stop('filemap length is unexpected!')
}
write.csv(filemap, file=output_path, row.names = FALSE)
message('Created ', output_path)

## Inspect filemap -------------------------------------------------------------
filemap = read.csv(output_path)
filemap$SerialNo   = factor(filemap$SerialNo)
filemap$UnitType   = factor(filemap$UnitType)
filemap$DeployNo   = factor(filemap$DeployNo)
filemap$DataYear   = factor(filemap$DataYear)
filemap$SurveyDate = as.POSIXlt(filemap$SurveyDate, tz = tz)

# Explore
unique(filemap$SerialNo)
unique(filemap$UnitType)
unique(filemap$DeployNo)
tapply(filemap$SurveyDate, filemap$SerialNo, unique)

# Show calendars of recorded dates
library(calendR)
for (year in unique(filemap$DataYear)) {
  message('Creating calendar for ', year)
  filemap_year = filemap[filemap$DataYear==year,]
  print(calendR(year = year,
          start_date = as.Date(filemap_year$SurveyDate)[1],
          end_date   = as.Date(filemap_year$SurveyDate[nrow(filemap_year)]),
          start = 'M'))
}
