library(vegan)
library(ggplot2)
library(ggrepel)
library(dplyr)
library(lubridate)
theme_set(theme_minimal())

# path to raw data from a site deployment (stand init from early June)
path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/raw_detections/2020/Deployment5/S4A04271_20200604'
setwd(path)

sitedate = basename(path)

files = list.files(pattern = "\\.csv$")

ndays = 1
files = files[1:(ndays*24)]

sample_sec_size = 3
sec_per_day = 60 * 60 * 24
samp_per_day = sec_per_day / sample_sec_size

# combine all data
data_list <- list()
for (f in files) {
  print(f)
  data_list[[f]] <- read.csv(f, row.names = NULL)
}
data <- do.call(rbind, data_list)
rm(data_list)

# convert to presence/absence with threshold
data = data[, c('common_name', 'confidence', 'start_date')]
# data$start_date = as.POSIXct(data$start_date)
threshold = 0.5
deets = data[,c('common_name', 'start_date')]
deets$presence = ifelse(data$confidence > threshold, 1, 0)
ref = reshape2::dcast(deets, start_date ~ common_name, value.var = 'presence')
ref$start_date = as.POSIXct(ref$start_date)

sunrise_time = as.POSIXct("2020-06-05 04:40:00")
sunset_time = as.POSIXct("2020-06-05 21:03:00")

get_times_by_interval_window = function(data_t, t_start, t_end, by = '1 min', window = 10) {
  interval_times = seq(from = t_start, to = t_end, by = by)
  nearest_times = as.POSIXct(unique(sapply(interval_times, function(x){ data_t[which.min(abs(x - data_t))] })))
  # Filter 'data_t' within 'window' seconds of each time in nearest_times
  times <- as.POSIXct(unlist(lapply(nearest_times, function(t) {
    data_t[data_t >= t & data_t < t + window]
  })))
  return(data_t[data_t %in% times])
}



########### show detection density
# df = deets
# df$start_date = as.POSIXct(df$start_date)
# interval <- 30 * 60  # 15 minutes in seconds
# df$bin = cut(df$start_date, breaks = seq(min(df$start_date), max(df$start_date) + interval, by = interval), include.lowest = TRUE)
# sum_by_interval <- df %>%
#   group_by(bin) %>%
#   summarise(total_presence = sum(presence))
# sum_by_interval = sum_by_interval[sum_by_interval$total_presence > 0, ]
# ggplot(sum_by_interval) + geom_point(mapping = aes(x = as.POSIXct(bin), y = total_presence)) + scale_x_datetime(date_breaks = "1 hour", date_labels = "%H")
###########


# SCHEMES for subsampling -----------------------

equal_24h_d2min_l60sec = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                              t_start = as.POSIXct("2020-06-05 00:00:00"),
                                                                              t_end = as.POSIXct("2020-06-05 23:59:59"),
                                                                              by = '2 min', window = 60), ]

equal_24h_d5min_l10sec = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                               t_start = as.POSIXct("2020-06-05 00:00:00"),
                                                                               t_end = as.POSIXct("2020-06-05 23:59:59"),
                                                                               by = '5 min', window = 10), ]

equal_24h_d3min_l6sec = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                              t_start = as.POSIXct("2020-06-05 00:00:00"),
                                                                              t_end = as.POSIXct("2020-06-05 23:59:59"),
                                                                              by = '3 min', window = 6), ]

equal_24h_d10min_l30sec = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                               t_start = as.POSIXct("2020-06-05 00:00:00"),
                                                                               t_end = as.POSIXct("2020-06-05 23:59:59"),
                                                                               by = '10 min', window = 30), ]

sunrise_m1p2_d5min_l10sec = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                                 t_start = sunrise_time - hours(1),
                                                                                 t_end = sunrise_time + hours(2),
                                                                                 by = '5 min', window = 10), ]
sunrise_m1p2_d1min_l10sec = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                                 t_start = sunrise_time - hours(1),
                                                                                 t_end = sunrise_time + hours(2),
                                                                                 by = '1 min', window = 10), ]


risetimes = get_times_by_interval_window(ref$start_date,
                                         t_start = sunrise_time - hours(1),
                                         t_end = sunrise_time + hours(2),
                                         by = '1 min', window = 10)
settimes = get_times_by_interval_window(ref$start_date,
                                        t_start = sunset_time - hours(1),
                                        t_end = sunset_time + hours(1),
                                        by = '1 min', window = 10)

riseandset_d1min_l10sec = ref[ref$start_date %in% c(risetimes, settimes), ]

focusrise_1m10stoSet = ref[ref$start_date %in% c(
  risetimes,
  get_times_by_interval_window(ref$start_date,
                               t_start = sunrise_time + hours(2),
                               t_end = sunset_time + hours(1),
                               by = '1 min', window = 10)
),]

testy = ref[ref$start_date %in% c(
  get_times_by_interval_window(ref$start_date,
                               t_start = sunrise_time - hours(1),
                               t_end = sunrise_time + hours(1),
                               by = '3 min', window = 60),
  get_times_by_interval_window(ref$start_date,
                               t_start = sunrise_time + hours(1),
                               t_end = sunset_time + hours(1),
                               by = '9 min', window = 30)
),]

baba = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                            t_start = sunrise_time,
                                                            t_end = sunset_time,
                                                            by = '5 min', window = 10), ]

baba_with_night = ref[ref$start_date %in% c(
  get_times_by_interval_window(ref$start_date,
                               t_start = as.POSIXct("2020-06-05 00:00:00"),
                               t_end = sunset_time,
                               by = '5 min', window = 10),
  get_times_by_interval_window(ref$start_date,
                               t_start = sunset_time,
                               t_end = as.POSIXct("2020-06-05 23:59:59"),
                               by = '5 min', window = 10)
), ]

baba_more_often = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                            t_start = sunrise_time,
                                                            t_end = sunset_time,
                                                            by = '3 min', window = 10), ]

baba_longer = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                       t_start = sunrise_time,
                                                                       t_end = sunset_time,
                                                                       by = '5 min', window = 20), ]

baba_more_often_and_longer = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                   t_start = sunrise_time,
                                                                   t_end = sunset_time,
                                                                   by = '4 min', window = 20), ]

schemes = list(
  # continuous full reference
  # cont_ref = ref,
  
  # continuous for certain hours
  # cont_04_07 = ref[format(ref$start_date, '%H') %in% c('04', '05', '06'), ],
  # cont_05_07 = ref[format(ref$start_date, '%H') %in% c('05', '06'), ],
  # the 1 hour before sunrise and 3 hours following
  # cont_sunrise_4h = ref %>% filter(start_date >= sunrise_time - hours(1) & start_date <= sunrise_time + hours(3)),
  # sunrise_m1p2_d5min_l10sec = sunrise_m1p2_d5min_l10sec,
  # sunrise_m1p2_d1min_l10sec = sunrise_m1p2_d1min_l10sec,
  # equal_24h_d10min_l30sec = equal_24h_d10min_l30sec,
  # equal_24h_d5min_l10sec = equal_24h_d5min_l10sec,
  # everyothermin = equal_24h_d2min_l60sec,
  # riseandset_d1min_l10sec = riseandset_d1min_l10sec,
  # focusrise_1m10stoSet = focusrise_1m10stoSet,
  # testy = testy,
  # equal_24h_d3min_l6sec = equal_24h_d3min_l6sec,
  baba = baba,
  baba_with_night = baba_with_night,
  baba_more_often = baba_more_often,
  baba_longer = baba_longer,
  baba_more_often_and_longer = baba_more_often_and_longer
)

df_compare = data.frame()
for (i in 1:length(schemes)) {
  scheme_name = names(schemes)[i]
  scheme = schemes[[i]]
  samp_in_hr = nrow(scheme) * 3 / 60 / 60
  print(paste('scheme', scheme_name, '( samp hr', samp_in_hr, ')'))
  scheme = scheme[, !names(scheme) %in% 'start_date']
  sp1 = specaccum(subset(scheme, select=-c(1)), method = 'collector')
  df = data.frame(samples = sp1$sites, richness = sp1$perm, scheme = scheme_name)
  df_compare = rbind(df_compare, df)
}

ggplot() +
  geom_line(df_compare, mapping = aes(x = samples, y = richness, group = scheme, color = scheme), alpha = 0.7) +
  labs(title = 'single day', subtitle = paste('threshold', threshold))


