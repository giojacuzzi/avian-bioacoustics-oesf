source('global.R')

data = get_site_strata_date_serial_data()

# 16 watersheds
watersheds = unique(data$WatershedID)
watersheds
length(watersheds)
# 4 strata
strata = unique(data$Strata)
strata
# 225 stations
stations = unique(data$StationName_AGG)
stations
length(stations)

# Some watersheds have all 4 strata, others do not
tapply(data$Strata, data$WatershedID, unique)
tapply(data$Strata, data$WatershedID, function(x) { length(unique(x)) })
# As such, each strata has a unique number of stations
tapply(data$StationName_AGG, data$Strata, function(x) { length(unique(x)) })

# Stations per watershed per strata
data_per_station = data[!duplicated(data$StationName_AGG),]
data_per_station %>% group_by(WatershedID, Strata) %>% summarise(NumStations=n(), .groups = 'drop') %>%
  as.data.frame()

# Subset data by year
data_2020 = data[data$DataYear==2020,]
data_2021 = data[data$DataYear==2021,]

# Number of stations per watershed per year
tapply(data_2020$StationName_AGG, data_2020$WatershedID, function(x) { length(unique(x)) })
tapply(data_2021$StationName_AGG, data_2021$WatershedID, function(x) { length(unique(x)) })

# Stations per watershed per strata per year
data_per_station_2020 = data_2020[!duplicated(data_2020$StationName_AGG),]
data_per_station_2020 %>% group_by(WatershedID, Strata) %>% summarise(NumStations=n(), .groups = 'drop') %>%
  as.data.frame()
data_per_station_2021 = data_2021[!duplicated(data_2021$StationName_AGG),]
data_per_station_2021 %>% group_by(WatershedID, Strata) %>% summarise(NumStations=n(), .groups = 'drop') %>%
  as.data.frame()

# 2020 mostly had survey lengths of 10 days, where every day was recorded
library(dplyr)
data_2020 %>% group_by(WatershedID, StationName_AGG) %>% summarise(n=n(), .groups = 'drop') %>%
  as.data.frame()
# 2021 mostly had survey lengths of 10 days, but where only 4 days of the 10 were recorded
data_2021 %>% group_by(WatershedID, StationName_AGG) %>% summarise(total_count=n(), .groups = 'drop') %>%
  as.data.frame()


