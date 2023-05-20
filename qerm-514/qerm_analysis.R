source('global.R')
library(ggplot2)
theme_set(theme_minimal())

# Get data from 2021
data = get_joined_data()
data_2021 = na.omit(data[year(data$SurveyDate)==2021,])

# Plot temperature data
ggplot(data_2021, aes(x = SurveyDate, y = TempMax, color = Strata)) +
  geom_point()

# Get ACI data
# TODO
