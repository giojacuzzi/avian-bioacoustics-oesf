library(xlsx)
library(dplyr)
library(stringr)
library(lubridate)

# Paths to directories comprising the DNR database
database_paths = c(
#  '~/../../Volumes/GIOJ Backup/DNR',
#  '~/../../Volumes/SAFS Backup/DNR'
  normalizePath('D:\\DNR', mustWork = T)
)
database_path_working = '~/../../Volumes/SAFS Work/DNR'

tz = 'America/Los_Angeles'

recording_date_serial_output_path = 'data/output/recording_date_serial_data.csv'

get_joined_data = function() {
  a = get_site_strata_date_serial_data()
  b = get_recording_date_serial_data()
  ab = full_join(a, b, by=c('SerialNo', 'SurveyDate','UnitType','DeployNo','DataYear'))
  c = get_site_date_temperature_data()
  abc = full_join(ab, c, by=c('SerialNo', 'SurveyDate'))
  return(abc)
}

## Read recording file, date, serial table from file
get_recording_date_serial_data = function() {
  
  message('Reading ', recording_date_serial_output_path)
  data = read.csv(recording_date_serial_output_path)
  # Format data
  data$SerialNo   = factor(data$SerialNo)
  data$UnitType   = factor(data$UnitType)
  data$DeployNo   = factor(data$DeployNo)
  data$DataYear   = factor(data$DataYear)
  data$SurveyDate = as.Date(data$SurveyDate)  #as.POSIXct(data$SurveyDate, tz=tz)
  data$DataTime   = as.POSIXct(data$DataTime, tz=tz)
  return(data)
}

get_site_date_temperature_data = function() {
  file = 'data/output/aru_temperature_data.csv'
  message('Reading ', file)
  data = read.csv(file)
  data$SerialNo   = factor(data$SerialNo)
  data$SurveyDate = as.Date(data$SurveyDate)
  return(data)
}

## Read date, site, strata, serial table from file
## Metadata:
# SurveyID        Identifier for the ARU deployment (i.e., period of time when an ARU was set in a site)
# SiteID          Identifier for the location
# DataYear        Year data were collected
# DeployNo        This is for finding a deployment datasheets (if needed), which were scanned in batches corresponding to batches of units deployment in a ~2-week period
# StationName     Unique identifier for the location
# StationName_AGG If a location was shifted between 2021 and 2022, name of the station/location the data are intended to represent (i.e., aggregate to)
# SurveyType      Acoustic or habitat survey
# SerialNo        ARU serial number
# SurveyDate      Date that audio data was downloaded for, in 1-hour files (Minis or SM4) or 5.5 hour files (SM2)
# DataHours       Number of hours in that day with audio data
# UnitType        Songmeter Minis (SMA), Songmeter SM4 (S4A), and Songmeter SM2 (SM2). The large majority of data were collected using Minis, the others were used only as needed to for round out deployments.
# Strata          Habitat type of the station
# UTM_E           Station Easting
# UTM_N           Station Northing

get_site_strata_date_serial_data = function() {
  
  file = 'data/ARUdates-site-strata_v2.xlsx'
  message('Reading ', file)
  data = read.xlsx(file, sheetName = 'ARUdates-site-strata')
  
  # Format data
  cols_to_factor = c('SurveyID', 'SiteID', 'DataYear', 'DeployNo', 'StationName', 'StationName_AGG', 'SurveyType', 'SerialNo', 'UnitType', 'Strata')
  data[cols_to_factor] = lapply(data[cols_to_factor], factor)
  data$SurveyDate = as.Date(data$SurveyDate) #as.POSIXct(data$SurveyDate, tz=tz, format='%d/%m/%Y')
  data$WatershedID = factor(substr(data$StationName_AGG, 1, 2))
  data$StationName_AGG = factor(data$StationName_AGG)
  data$StationName = factor(data$StationName)
  
  return(data)
}

## Additional info:
# Sample rate was 32 kHz (or it was supposed to be!) on everything
# Gain settings were 16 dB for SM2 and SM4s and 18 dB for Minis
# (according to Wildlife Acoustics those are equivalent gain settings
# across the different units).
# Everything was channel left, although there was one deployment where some
# got recorded in stereo by accident (no mic, though, so it's empty audio).


get_serial_from_file_name = function(file) {
  substrings = str_split(str_sub(basename(file), start = 1, end = -5), '_')
  return(sapply(substrings, '[[', 1))
}

get_date_from_file_name = function(file) {
  substrings = str_split(str_sub(basename(file), start = 1, end = -5), '_')
  date_raw = sapply(substrings, '[[', 2)
  date = ymd(date_raw)
  return(date)
}

get_time_from_file_name = function(file) {
  substrings = str_split(str_sub(basename(file), start = 1, end = -5), '_')
  date_raw = sapply(substrings, '[[', 2)
  year = substring(date_raw, 1, 4)
  date = ymd(date_raw) #as.POSIXct(date_raw, tz = tz, format='%Y%m%d')
  time = as.POSIXct(
    paste(date_raw, sapply(substrings, '[[', 3)),
    tz = tz, format='%Y%m%d %H%M%S')
  return(time)
}

get_hour_from_file_name = function(file) {
  substrings = str_split(str_sub(basename(file), start = 1, end = -5), '_')
  date_raw = sapply(substrings, '[[', 2)
  year = substring(date_raw, 1, 4)
  date = ymd(date_raw) #as.POSIXct(date_raw, tz = tz, format='%Y%m%d')
  time = as.POSIXct(
    paste(date_raw, sapply(substrings, '[[', 3)),
    tz = tz, format='%Y%m%d %H%M%S')
  hour = format(round(time, units='hours'), format='%H')
  return(hour)
}
