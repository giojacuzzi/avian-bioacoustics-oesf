source('global.R')
library(ggplot2)
theme_set(theme_minimal())

# Get data from 2020, 2021
data = get_joined_data()

data = na.omit(data[data$DataYear %in% c(2020,2021), ])

# Plot temperature data
ggplot(data, aes(x = SurveyDate, y = TempMax, color = Strata)) +
  geom_point() + labs(title = 'Temperature per strata')

# Temperature stats by strata
sort(tapply(data$TempMax, data$Strata, median))
sort(tapply(data$TempMax, data$Strata, mean))

# Get index data
data_acidx = read.csv('acoustic_indices/output/solar_noon/results.csv')
data_acidx$SurveyDate = as.Date(data_acidx$SurveyDate)
data_acidx$SerialNo = factor(data_acidx$SerialNo)
data_acidx = left_join(data_acidx,
                       distinct(data[, c('SurveyDate', 'SerialNo', 'Strata')])
                       , by = c('SurveyDate', 'SerialNo'))

# Plot index data
ggplot(data_acidx, aes(x = SurveyDate, y = ACI, color = Strata)) +
  geom_point() + labs(title = 'ACI per strata')

# Get stats by strata
sort(tapply(data_acidx$ACI, data_acidx$Strata, median))
sort(tapply(data_acidx$BIO, data_acidx$Strata, median))
