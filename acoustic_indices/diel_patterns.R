source('acoustic_indices/batch_process_alpha_indices.R')
source('global.R')

site_data = get_joined_data()

# Get all units active on a particular SurveyDate in a particular Strata
SurveyDate = '2020-04-19'

# Option: active_serialnos = unique(site_data[site_data$SurveyDate==SurveyDate,c('SerialNo','Strata')])
# Option: Get the first serial number of each strata (~40 minutes for 24 files)
# first_per_strata = sapply(tapply(active_serialnos$SerialNo, active_serialnos$Strata, function(x) { x }), '[[', 1)

# Take the files from the full database drives
# files = site_data[site_data$SurveyDate==SurveyDate &
#                     (site_data$NearHour>= HourInterval[1] &
#                        site_data$NearHour<= HourInterval[2]) &
#                     site_data$SerialNo %in% active_serialnos$SerialNo, 'File']

# Option: Alternatively, take files from SAFS Work drive (Deployment1_April8_12, Deployment4_May20_24)
outpath = '~/../../Volumes/SAFS Work/DNR/2020/Deployment1_April8_12/_output/'
files = list.files(path='~/../../Volumes/SAFS Work/DNR/2020/Deployment1_April8_12', pattern="*.wav", full.names=T, recursive=T)
to_process = data.frame(
  File=files,
  SerialNo=site_data[match(basename(files), basename(site_data$File)), 'SerialNo'],
  Strata=site_data[match(basename(files), basename(site_data$File)), 'Strata'],
  SurveyDate=site_data[match(basename(files), basename(site_data$File)), 'SurveyDate'],
  NearHour=site_data[match(basename(files), basename(site_data$File)), 'NearHour'],
  DataTime=site_data[match(basename(files), basename(site_data$File)), 'DataTime']
)

# Only process specific hours
HourInterval = c(0,23) # inclusive
to_process = to_process[(to_process$NearHour >= HourInterval[1] &
                           to_process$NearHour <= HourInterval[2]),]

# Alpha acoustic index arameters
indices = c('ACI','BIO','ADI', 'H', 'NDSI')
interval = 60 * 2 # in sec (i.e. seconds * minutes)
thresh_db = -50
min_f = 1800
max_f = 9000
step_f = 1000
window = 512

# View spectrogram of an hour
# spectrogram(readWave(files[4]), alim = c(db_threshold, 0))

#####
# Pick up where we left off...
if (exists('results_files')) {
  to_process = to_process[!(get_serial_from_file_name(basename(to_process$File)) %in% get_serial_from_file_name(basename(results_files))), ]
}
#####

########## PROCESS ##########
i = 1
for (serial in unique(to_process$SerialNo)) {
  
  outfile = paste0(serial,'_',SurveyDate) #'diel_patterns' #SurveyDate #paste(Strata, unit)
  infiles = to_process[to_process$SerialNo==serial,'File']
  
  message('Processing ', serial, ' (', i, ' of ', length(unique(to_process$SerialNo)),' sites)...')

  batch_process_alpha_indices(
    input_files   = infiles,
    output_path   = outpath,
    output_file   = outfile,
    alpha_indices = indices,
    time_interval = interval,
    ncores        = 8,
    dc_correct    = T,
    digits        = 4,
    diagnostics   = T,
    # Alpha index parameters
    min_freq = min_f,
    max_freq = max_f,
    anthro_min = 10,
    anthro_max = min_f, 
    bio_min = min_f,
    bio_max = max_f,
    db_threshold = thresh_db, 
    freq_step = step_f,
    j = 5,
    fft_w = window,
    wl = window
  )
  i = i+1
}
##############################

# Fetch results
results_files    = list.files(path=outpath, pattern="*.csv", full.names=T, recursive=F)
diagnostic_files = results_files[grepl('diagnostics', results_files, fixed = T)]
results_files    = results_files[!grepl('diagnostics', results_files, fixed = T)]

source('helpers.R')
library(ggplot2)
theme_set(theme_minimal())

# Explore diagnostics
diagnostics = data.frame()
for (d in diagnostic_files) {
  diagnostics = rbind(diagnostics, read.csv(d, header=T))
}
summary(diagnostics$Clipping)
summary(linear_to_dBFS(diagnostics$DcBias))
ggplot(diagnostics, aes(x=linear_to_dBFS(DcBias))) + geom_histogram()

# Explore data
data = data.frame()
for (r in results_files) {
  data = rbind(data, read.csv(r, header=T))
}
test = site_data

test$BaseFile = basename(test$File)
data$BaseFile = basename(data$File)
data = left_join(data, test, by = c('BaseFile'))
data$DataTime = as.POSIXct(data$DataTime, tz=tz) + data$Start # calculate actual start times of each interval
data$Strata = factor(data$Strata, levels=c('STAND INIT', 'COMP EXCL', 'MATURE', 'THINNED')) # levels = c("small", "medium", "large")
data$SerialNo = factor(data$SerialNo)
summary(data$Strata)

# Average data per a time interval cut
data$CutHour = cut(data$DataTime, '5 min') # e.g. '1 hour'

# Plot all SerialNo for a specific Strata
t_start = as.POSIXct('2020-04-19 00:00:00 PDT')
t_end   = as.POSIXct('2020-04-20 00:00:00 PDT')
t_sunrise = as.POSIXct('2020-04-19 06:19:00 PDT')
t_sunset  = as.POSIXct('2020-04-19 20:15:00 PDT')
strata_colors = c('#A4BE7B', '#A25B5B', '#285430', '#54436B')
means_serialno = data.frame(
  summarise(group_by(data, Strata, SerialNo, CutHour),
            ACI=mean(ACI), BIO=mean(BIO), ADI=mean(ADI),
            AEI=mean(AEI), H=mean(H), M=mean(M), NDSI=mean(NDSI))
)
means_serialno$CutHour = as.POSIXct(means_serialno$CutHour)
means_serialno = means_serialno[means_serialno$CutHour >= t_start & # limit to 24 hour window
                                  means_serialno$CutHour < t_end,]

ggplot(means_serialno[means_serialno$Strata=='STAND INIT',], aes(x=CutHour, y=NDSI, color=SerialNo)) +
  geom_line(linetype=3) +
  geom_smooth(aes(fill=SerialNo), method='loess', se=T, span=0.2, linewidth=1, alpha=0.15) +
  scale_x_datetime(date_breaks = '2 hours', date_labels = '%H', limits=c(t_start,t_end))

# Plot all averaged Strata
means = data.frame(
  summarise(group_by(data, Strata, CutHour),
            ACI=mean(ACI), BIO=mean(BIO), ADI=mean(ADI),
            AEI=mean(AEI), H=mean(H), M=mean(M), NDSI=mean(NDSI))
)
levels(means$Strata) = c('Stand initiation', 'Competitive exclusion', 'Mature', 'Commercially thinned') # rename factor levels
means$CutHour = as.POSIXct(means$CutHour)
means = means[means$CutHour >= t_start & # limit to 24 hour window
                means$CutHour < t_end,]
ggsave_w = 15
ggsave_h = 7

p_aci = ggplot(means, aes(x=CutHour, y=ACI, color=Strata)) +
  # geom_point(size=0.2) +
  geom_vline(xintercept=t_sunrise, linetype='dotted', alpha=0.7) +
  geom_vline(xintercept=t_sunset,  linetype='dotted', alpha=0.7) +
  geom_smooth(aes(fill=Strata), method='loess', se=T, span=0.2, linewidth=1, alpha=0.15) + 
  scale_color_manual(values=strata_colors) + scale_fill_manual(values=strata_colors) +
  scale_x_datetime(date_breaks = '2 hours', date_labels = '%H', limits=c(t_start,t_end), expand=c(0,0)) +
  labs(title='Acoustic Complexity Index', x='Hour' ,y='') + theme(legend.position='none') + guides(color = guide_legend(''), fill = 'none')
print(p_aci)
ggsave(p_aci + theme(text = element_text(size = 30), plot.margin = margin(1,1,1,1, 'cm')),
       file='~/Desktop/p_aci.png', width=ggsave_w, height=ggsave_h)

p_adi = ggplot(means, aes(x=CutHour, y=ADI, color=Strata)) +
  # geom_point(size=0.2) +
  geom_vline(xintercept=t_sunrise, linetype='dotted', alpha=0.7) +
  geom_vline(xintercept=t_sunset,  linetype='dotted', alpha=0.7) +
  geom_smooth(aes(fill=Strata), method='loess', se=T, span=0.3, linewidth=1, alpha=0.15) + 
  scale_color_manual(values=strata_colors) + scale_fill_manual(values=strata_colors) +
  scale_x_datetime(date_breaks = '2 hours', date_labels = '%H', limits=c(t_start,t_end), expand=c(0,0)) +
  labs(title='Acoustic Diversity Index', x='Hour', y='') + theme(legend.position='none') + guides(fill = 'none')
print(p_adi)
ggsave(p_adi + theme(text = element_text(size = 30), plot.margin = margin(1,1,1,1, 'cm')), file='~/Desktop/p_adi.png', width=ggsave_w, height=ggsave_h)

p_ndsi = ggplot(means, aes(x=CutHour, y=NDSI, color=Strata)) +
  # geom_point(size=0.2) +
  geom_hline(yintercept=0, linetype='solid') +
  geom_vline(xintercept=t_sunrise, linetype='dotted', alpha=0.7) +
  geom_vline(xintercept=t_sunset,  linetype='dotted', alpha=0.7) +
  geom_smooth(aes(fill=Strata), method='loess', se=T, span=0.3, linewidth=1, alpha=0.15) + 
  scale_color_manual(values=strata_colors) + scale_fill_manual(values=strata_colors) +
  scale_x_datetime(date_breaks = '2 hours', date_labels = '%H', limits=c(t_start,t_end), expand=c(0,0)) +
  labs(title='Normalized Difference Soundscape Index', x='Hour', y='') + theme(legend.position='none') + guides(fill = 'none')
print(p_ndsi)
ggsave(p_ndsi + theme(text = element_text(size = 30), plot.margin = margin(1,1,1,1, 'cm')), file='~/Desktop/p_ndsi.png', width=ggsave_w, height=ggsave_h)

# Distributions during 1h before and after dawn to dusk
# means_day = means[means$CutHour >= t_sunrise - 60*60 & # limit to 24 hour window
                # means$CutHour < t_end + 60*60,]
 ggplot(means, aes(x=Strata, y=ACI, fill=Strata)) +
  geom_boxplot() + # outlier.shape = NA)
  scale_fill_manual(values=strata_colors) +
  theme(legend.position='none')
   #+ coord_cartesian(ylim = c(1620,1850))
ggplot(means, aes(x=Strata, y=ADI, fill=Strata)) +
  geom_boxplot() + #outlier.shape = NA
  scale_fill_manual(values=strata_colors) +
  theme(legend.position='none')
#+ coord_cartesian(ylim = c(2.025,2.2))
ggplot(means, aes(x=Strata, y=NDSI, fill=Strata)) +
  geom_boxplot() + scale_fill_manual(values=strata_colors) +
  theme(legend.position='none')


# data %>% 
#   group_by(SerialNo) %>%
#   mutate(maxtime = max(DataTime)) %>%
#   group_by(Strata) %>%
#   mutate(maxtime = min(maxtime)) %>%
#   group_by(SerialNo, Strata, DataTime) %>%
#   summarize(ACI = mean(ACI), .groups=c(SerialNo,Strata)) %>%
#   ggplot(aes(DataTime, ACI, colour = Strata)) + geom_line()

# TODO: fit to time?
library(tidyr)
library(lubridate)
test = na.omit(data)
test = test %>% group_by(SerialNo) %>% complete(DataTime = seq(floor_date(min(DataTime), unit = 'hours'), ceiling_date(max(DataTime), unit = 'hours'), by = 'sec')) %>% fill(names(test))
# test_ts = ts(test)

ggplot(test, aes(x=DataTime, y=ACI, color=Strata, linetype=SerialNo)) + geom_line()
test %>% group_by(Strata, DataTime) %>% summarize(ACI = mean(ACI)) %>% ggplot(aes(DataTime, ACI, colour = Strata)) + geom_line()

testy = test %>% group_by(Strata, DataTime) %>% summarize(ACI = mean(ACI))
testy = testy[!duplicated(testy$ACI),]
ggplot(testy, aes(x=DataTime, y=ACI, color=Strata)) + geom_line()


ggplot(data, aes(x=DataTime, y=ACI, color=Strata, linetype=SerialNo)) + geom_line()
ggplot(data, aes(x=DataTime, y=BIO, fill=Strata)) + geom_boxplot()

# TODO: ts? aggregate?

# Apply rolling mean to data to smooth
# library(zoo)
# k = interval / (10) # width of rolling window (interval/x sec)
# mean_data <- data %>%
#   group_by(Strata) %>% 
#   mutate(ACI_mean = rollmean(ACI, k, na.pad = T)) %>%
#   mutate(ADI_mean = rollmean(ADI, k, na.pad = T)) %>%
#   mutate(AEI_mean = rollmean(AEI, k, na.pad = T)) %>%
#   mutate(BIO_mean = rollmean(BIO, k, na.pad = T)) %>%
#   mutate(H_mean = rollmean(H, k, na.pad = T))
# ggplot(mean_data, aes(x=Time, y=ACI_mean, color=Strata)) + geom_line() + labs(title = 'ACI')
# ggplot(mean_data, aes(x=Time, y=ADI_mean, color=Strata)) + geom_line() + labs(title = 'ADI')
# ggplot(mean_data, aes(x=Time, y=AEI_mean, color=Strata)) + geom_line() + labs(title = 'AEI')
# ggplot(mean_data, aes(x=Time, y=BIO_mean, color=Strata)) + geom_line() + labs(title = 'BIO')
# ggplot(mean_data, aes(x=Time, y=H_mean, color=Strata)) + geom_line() + labs(title = 'H')
