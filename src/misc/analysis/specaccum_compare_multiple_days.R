library(vegan)
library(ggplot2)
library(ggrepel)
library(dplyr)
library(lubridate)
theme_set(theme_minimal())

# path to raw data from a site deployment (stand init from early June)
# path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/raw_detections/2020/Deployment5/S4A04325_20200603' # MATURE
path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/raw_detections/2020/Deployment5/S4A04271_20200604' # STAND INIT
setwd(path)

sitedate = basename(path)

files = list.files(pattern = "\\.csv$")

ndays = 10
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
threshold = 0.75
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

cont_1d1 = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                              t_start = as.POSIXct("2020-06-05 00:00:00"),
                                                                              t_end = as.POSIXct("2020-06-05 23:59:59"),
                                                                              by = '1 min', window = 60), ]
cont_1d2 = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                               t_start = as.POSIXct("2020-06-06 00:00:00"),
                                                               t_end = as.POSIXct("2020-06-06 23:59:59"),
                                                               by = '1 min', window = 60), ]
cont_1d3 = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                t_start = as.POSIXct("2020-06-07 00:00:00"),
                                                                t_end = as.POSIXct("2020-06-07 23:59:59"),
                                                                by = '1 min', window = 60), ]
cont_1d4 = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                t_start = as.POSIXct("2020-06-08 00:00:00"),
                                                                t_end = as.POSIXct("2020-06-08 23:59:59"),
                                                                by = '1 min', window = 60), ]
cont_1d5 = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                t_start = as.POSIXct("2020-06-09 00:00:00"),
                                                                t_end = as.POSIXct("2020-06-09 23:59:59"),
                                                                by = '1 min', window = 60), ]

cont_night = ref[ref$start_date %in% c(get_times_by_interval_window(ref$start_date,
                                                                    t_start = as.POSIXct("2020-06-05 00:00:00"),
                                                                    t_end = sunrise_time,
                                                                    by = '1 min', window = 60),
                                       get_times_by_interval_window(ref$start_date,
                                                               t_start = sunset_time,
                                                               t_end = as.POSIXct("2020-06-05 23:59:59"),
                                                               by = '1 min', window = 60)), ]

equal_24h_d2min_l60sec = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                              t_start = as.POSIXct("2020-06-05 00:00:00"),
                                                                              t_end = as.POSIXct("2020-06-14 23:59:59"),
                                                                              by = '2 min', window = 60), ]

eq_1d_5m_10s = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                            t_start = sunrise_time + 86400 * 1,
                                                            t_end = sunset_time + 86400 * 1,
                                                            by = '5 min', window = 10), ]

eq_2d_5m_10s = ref[ref$start_date %in% c(
  get_times_by_interval_window(ref$start_date,
                               t_start = sunrise_time + 86400 * 1,
                               t_end = sunset_time  + 86400 * 1,
                               by = '5 min', window = 10),
  get_times_by_interval_window(ref$start_date,
                               t_start = sunrise_time + 86400 * 5,
                               t_end = sunset_time + 86400 * 5,
                               by = '5 min', window = 10)
), ]

eq_3d_5m_10s = ref[ref$start_date %in% c(
  get_times_by_interval_window(ref$start_date,
                               t_start = sunrise_time + 86400 * 1,
                               t_end = sunset_time + 86400 * 1,
                               by = '5 min', window = 10),
  get_times_by_interval_window(ref$start_date,
                               t_start = sunrise_time + 86400 * 5,
                               t_end = sunset_time + 86400 * 5,
                               by = '5 min', window = 10),
  get_times_by_interval_window(ref$start_date,
                               t_start = sunrise_time + 86400 * 9,
                               t_end = sunset_time + 86400 * 9,
                               by = '5 min', window = 10)
), ]


# eq_10d_5m_15s_prepost = 
# window = 15
# times = seq(from = sunrise_time - 3600, to = sunset_time + 3600, by = '5 min')
eq_1d_5m_15s = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                    t_start = sunrise_time + 86400 * 1,
                                                                    t_end = sunset_time + 86400 * 1,
                                                                    by = '5 min', window = 15), ]
eq_1d_5m_15s_buffer = ref[ref$start_date %in% get_times_by_interval_window(ref$start_date,
                                                                    t_start = sunrise_time + 86400 * 1 - 3600,
                                                                    t_end = sunset_time + 86400 * 1 + 3600,
                                                                    by = '5 min', window = 15), ]


#####

# Shaw et al 2020
shaw_eq_6m_60s = ref[ref$start_date %in% c(
  get_times_by_interval_window(ref$start_date,
                               t_start = sunrise_time + 86400 * 1 + 3600 * 3,
                               t_end = sunrise_time  + 86400 * 1 + 3600 * 4,
                               by = '6 min', window = 60),
  get_times_by_interval_window(ref$start_date,
                               t_start = sunrise_time + 86400 * 5 + 3600 * 3,
                               t_end = sunrise_time + 86400 * 5 + 3600 * 4,
                               by = '6 min', window = 60)
), ]

shaw_eq_1d_sunrise = ref[ref$start_date %in% c(
  get_times_by_interval_window(ref$start_date,
                               t_start = sunrise_time + 86400 * 1 + 3600 * 3,
                               t_end = sunrise_time  + 86400 * 1 + 3600 * 4,
                               by = '6 min', window = 60)
), ]
shaw_eq_1d_both = ref[ref$start_date %in% c(
  get_times_by_interval_window(ref$start_date,
                               t_start = sunrise_time + 86400 * 1 + 3600 * 3,
                               t_end = sunrise_time  + 86400 * 1 + 3600 * 4,
                               by = '6 min', window = 60),
  get_times_by_interval_window(ref$start_date,
                               t_start = sunset_time + 86400 * 1 - 3600 / 2,
                               t_end = sunset_time  + 86400 * 1 + 3600 / 2,
                               by = '6 min', window = 60)
), ]

shaw_eq_1d_cont = ref[ref$start_date %in% c(
  get_times_by_interval_window(ref$start_date,
                               t_start = as.POSIXct(format(sunrise_time, '%Y-%m-%d')) + 86400,
                               t_end = as.POSIXct(format(sunrise_time, '%Y-%m-%d')) + 86400 * 2 - 1,
                               by = '6 min', window = 60)
), ]

# Random across all days, with same times but distributed
start_date = as.POSIXct("2020-06-05")
end_date = as.POSIXct("2020-06-14")

window = 10
times = seq(from = sunrise_time, to = sunset_time, by = '5 min')
rdates = as.POSIXct(runif(length(times), start_date, end_date)) # Generate random dates within the specified range
random_times = as.POSIXct(paste(format(rdates, '%Y-%m-%d'), format(times, '%H:%M:%S')))
data_t = ref$start_date
nearest_times = as.POSIXct(unique(sapply(random_times, function(x){ data_t[which.min(abs(x - data_t))] })))
good_times <- as.POSIXct(unlist(lapply(nearest_times, function(t) { # expand window
  data_t[data_t >= t & data_t < t + window]
})))
rand_10d_5m_10s = ref[ref$start_date %in% good_times, ]


###
eq_5d_5m_10s = seq(from = start_date, to = end_date, by = "2 days")
resultz = as.POSIXct(paste(rep(format(eq_5d_5m_10s, '%Y-%m-%d'), each = length(times)), rep(format(times, '%H:%M:%S'), length(eq_5d_5m_10s))))
nearest_times = as.POSIXct(unique(sapply(random_times, function(x){ data_t[which.min(abs(x - data_t))] })))
good_times <- as.POSIXct(unlist(lapply(nearest_times, function(t) { # expand window
  data_t[data_t >= t & data_t < t + window]
})))
eq_5d_5m_10s = ref[ref$start_date %in% good_times, ]
###


# Random across all days, with same times but distributed
window = 10
times = seq(from = sunrise_time, to = sunset_time, by = '3 min')
random_times = as.POSIXct(paste(format(rdates, '%Y-%m-%d'), format(times, '%H:%M:%S')))
data_t = ref$start_date
nearest_times = as.POSIXct(unique(sapply(random_times, function(x){ data_t[which.min(abs(x - data_t))] })))
good_times <- as.POSIXct(unlist(lapply(nearest_times, function(t) { # expand window
  data_t[data_t >= t & data_t < t + window]
})))
rand_10d_3m_10s = ref[ref$start_date %in% good_times, ]

# Random across all days, with same times but distributed
window = 15
times = seq(from = sunrise_time, to = sunset_time, by = '5 min')
random_times = as.POSIXct(paste(format(rdates, '%Y-%m-%d'), format(times, '%H:%M:%S')))
data_t = ref$start_date
nearest_times = as.POSIXct(unique(sapply(random_times, function(x){ data_t[which.min(abs(x - data_t))] })))
good_times <- as.POSIXct(unlist(lapply(nearest_times, function(t) { # expand window
  data_t[data_t >= t & data_t < t + window]
})))
rand_10d_5m_15s = ref[ref$start_date %in% good_times, ]

# Random with additional times
window = 15
times = seq(from = sunrise_time - 3600, to = sunset_time + 3600, by = '5 min')
random_times = as.POSIXct(paste(format(rdates, '%Y-%m-%d'), format(times, '%H:%M:%S')))
data_t = ref$start_date
nearest_times = as.POSIXct(unique(sapply(random_times, function(x){ data_t[which.min(abs(x - data_t))] })))
good_times <- as.POSIXct(unlist(lapply(nearest_times, function(t) { # expand window
  data_t[data_t >= t & data_t < t + window]
})))
rand_10d_5m_15s_buffer = ref[ref$start_date %in% good_times, ]

schemes = list(
  # continuous full reference
  # cont_ref = ref,
  # everyothermin = equal_24h_d2min_l60sec,
  # eq_1d_5m_10s = eq_1d_5m_10s,
  # eq_2d_5m_10s = eq_2d_5m_10s,
  # eq_3d_5m_10s = eq_3d_5m_10s,
  # rand_10d_5m_10s = rand_10d_5m_10s,
  # rand_10d_3m_10s = rand_10d_3m_10s,
  # rand_10d_5m_15s = rand_10d_5m_15s,
  # eq_5d_5m_10s = eq_5d_5m_10s,
  shaw_eq_6m_60s = shaw_eq_6m_60s,
  cont_1d1 = cont_1d1,
  cont_1d2 = cont_1d2,
  cont_1d3 = cont_1d3,
  cont_1d4 = cont_1d4,
  cont_1d5 = cont_1d5,
  # rand_10d_5m_15s_buffer = rand_10d_5m_15s_buffer,
  eq_1d_5m_15s = eq_1d_5m_15s,
  eq_1d_5m_15s_buffer = eq_1d_5m_15s_buffer,
  # shaw_eq_1d_sunrise = shaw_eq_1d_sunrise,
  # shaw_eq_1d_both = shaw_eq_1d_both,
  shaw_eq_1d_cont = shaw_eq_1d_cont
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
  geom_line(df_compare, mapping = aes(x = samples, y = richness, group = scheme, color = scheme), alpha = 0.7, linewidth = 1) +
  labs(title = '', subtitle = paste('threshold', threshold))


