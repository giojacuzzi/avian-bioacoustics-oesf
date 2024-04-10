library(vegan)
library(ggplot2)
library(ggrepel)
library(tidyverse)
theme_set(theme_minimal())

# path to raw data from a site deployment (stand init from early June)
path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/raw_detections/2020/Deployment5/S4A04271_20200604'
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
  data_list[[f]] <- read_csv(f, show_col_types=FALSE)
}
data <- do.call(rbind, data_list)
rm(data_list)

data = data[, c('common_name', 'confidence', 'start_date')]

thresholds = c(0.5, 0.75, 0.9)

df_compare = data.frame()

for (threshold in thresholds) {
  print(paste('calculating threshold', threshold))
  
  deets = data[,c('common_name', 'start_date')]
  deets$presence = ifelse(data$confidence > threshold, 1, 0)
  
  pivoted_df <- reshape2::dcast(deets, start_date ~ common_name, value.var = "presence")
  pivoted_df = subset(pivoted_df, select=-c(1))
  
  sp1 = specaccum(pivoted_df, method = 'collector')
  fit = lm(sp1$perm~log(sp1$sites))
  df = data.frame(samples = sp1$sites, richness = sp1$perm, threshold = threshold, regression = predict(fit))
  df_compare = rbind(df_compare, df)
}
df_compare$threshold = factor(df_compare$threshold)
daily_breaks = seq(0, nrow(df), by = samp_per_day)

ggplot() +
  geom_line(df_compare, mapping = aes(x = samples, y = richness, group = threshold, color = threshold)) +
  geom_line(df_compare, mapping = aes(x = samples, y = regression, group = threshold), color = 'black', alpha = 0.5) +
  ylim(0, NA) +
  scale_x_continuous(breaks = daily_breaks, labels = seq(1,length(daily_breaks))) +
  labs(title = paste(sitedate, '-', ndays, 'days continuous'))
