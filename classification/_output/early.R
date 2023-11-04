source('global.R')
library(dplyr)
library(tidyr)
library(ggplot2)
library(viridis)
library(ggpattern)
library(vegan)
library(FactoMineR)
library (factoextra)
library(fossil)
source("~/Dropbox/Research/ESA_Julian_Analysis/biostats.R", encoding = 'UTF-8')

stage_colors  = c('#73E2A7', '#1C7C54', '#6c584c')
strata_colors = c('#73E2A7', '#1C7C54',  '#73125A', '#6c584c')
gradient.cols = gradient.cols = c("#00AFBB22", "#E7B800", "#FC4E07")
strata_labels = c('Early', 'Mid', 'Mid (Thinned)', 'Late')

## Preprocess data

in_dir = 'classification/_output/birdnet'
out_dir = 'classification/_output'

file_list <- list.files(path = in_dir, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
combined_data <- do.call(rbind, lapply(file_list, read.csv, colClasses = c(
  'common_name' = 'factor',
  'scientific_name' = 'factor',
  # 'start_date' = 'POSIXct',
  # 'end_date' = 'POSIXct',
  'serial_no' = 'factor',
  'file' = 'factor'
)))

d50 = combined_data[combined_data$confidence>0.5,]
d75 = combined_data[combined_data$confidence>=0.75,]
d95 = combined_data[combined_data$confidence>=0.95,]

length(unique(d50$common_name))
length(unique(d75$common_name))
length(unique(d95$common_name))

sort(summary(d50$common_name), decreasing=T)[1:10]
sort(summary(d75$common_name), decreasing=T)[1:10]
sort(summary(d95$common_name), decreasing=T)[1:10]

sort(summary(d50$common_name))[1:10]
sort(summary(d75$common_name))[1:10]
sort(summary(d95$common_name))[1:10]

# Only consider detections with a high enough confidence
# min_confidence = 0.95
data = d50

# NOTE: original dates are incorrect!
data$date = get_date_from_file_name(as.character(data[,'file'])) #as.Date(data$start_date)
site_data = get_site_strata_date_serial_data()

data = full_join(data, site_data[,c('serial_no', 'date', 'deploy', 'site', 'watershed', 'strata', 'stage', 'thinned')], by = c('serial_no', 'date'))
data = na.omit(data)
data$common_name = factor(data$common_name) # reduce factor levels

# Complex Early Seral Tag
data$strata = as.character(data$strata)
data[(data$watershed %in% c('Az', 'Bz', 'Cz', 'Dz') & data$strata == 'Stand Initiation'), 'strata'] = 'Complex Early'
data$strata = factor(data$strata)

sort(summary(data$common_name))[1:10]

# TODO: remove species that were only observed at a site on a single day in a given deployment
# Find the number of times each species was observed at each site during each deployment
unique_date_counts <- data %>%
  group_by(common_name, site, deploy, date) %>%
  summarize(observation_count = n_distinct(serial_no)) %>%
  filter(observation_count > 0) %>%
  group_by(common_name, site, deploy) %>%
  summarize(unique_date_count = n()) %>% filter(unique_date_count <= 1)
filtered_dataframe <- data %>%
  anti_join(unique_date_counts, by = c("common_name", "site", "deploy"))
data = filtered_dataframe

## Export total unique detections for multivariate ordination analysis
sort(as.character(unique(data$date)))
observations_per_date_site = data %>% select(date, site, strata, common_name) %>% group_by(date, site, strata, common_name) %>% summarise(count = n())
factors = c('site', 'strata', 'common_name')
observations_per_date_site[factors] = lapply(observations_per_date_site[factors], factor)
observations_per_date_site = as.data.frame(observations_per_date_site)

# write.csv(observations_per_date_site, 'classification/_output/_birdcounts_oesf.csv', row.names = F)

# Create full matrix of all species per site (aggregating dates), including nondetections (count == 0)
library(fossil)
m = create.matrix(observations_per_date_site, tax.name = 'common_name', locality = 'site', abund.col = 'count', abund = T)
n_dates_per_site = as.data.frame(tapply(observations_per_date_site$date, observations_per_date_site$site, function(x) { length(unique(x))}))
n_dates_per_site$site = rownames(n_dates_per_site)
colnames(n_dates_per_site) = c('n_dates', 'site')

# Average detections across deployment dates
test = m
df_divisor = t(n_dates_per_site['n_dates'])
for (col_name in colnames(test)) {
  print(col_name)
  if (col_name %in% colnames(df_divisor)) {
    test[, col_name] = test[, col_name] / df_divisor['n_dates', col_name]
  }
}
mat = apply(test, c(1, 2), round) # round
mat <- apply(test, c(1, 2), function(x) { ifelse(x < 1, 0, round(x)) }) # remove less than one average values (assumed transient detections), and round

sites_to_strata = distinct(select(site_data, site, strata))
sites_to_strata = with(sites_to_strata, sites_to_strata[order(as.character(site)), ])
sites_to_strata = sites_to_strata[sites_to_strata$site %in% colnames(m), ]
sites_to_strata$strata = as.character(sites_to_strata$strata)
sites_to_strata[(grepl('z', sites_to_strata$site, fixed=T) & sites_to_strata$strata == 'Stand Initiation'), 'strata'] = 'Complex Early'
sites_to_strata$strata = factor(sites_to_strata$strata)

###### PCA
sitedata = t(mat)
sitedata <- drop.var(sitedata,min.po=1)

# Set to presence/absence 1 or 0
# sitedata[-1] <- as.integer(sitedata[-1] > 1) # TODO: relab or p/a?
sitedata<-decostand(sitedata, method='hel')
bird.pca<-PCA(sitedata,graph=F)

theme_set(theme_minimal())

fviz_screeplot(bird.pca, addlabels = TRUE, ylim = c(0, 30))
fviz_pca_var(bird.pca, col.var="contrib", gradient.cols = gradient.cols, repel = F, title="PCA variables (species)", ggtheme = theme(text = element_text(size = 18)))

fviz_pca_biplot(bird.pca, label="var",col.var="contrib",repel = F, title="", ggtheme = theme(text = element_text(size = 18)))

fviz_pca_ind(bird.pca, geom = "point", habillage = sites_to_strata$strata,  
             addEllipses = TRUE, ellipse.level=0.95,
             title="", ggtheme = theme(text = element_text(size = 18)))

# PERMANOVA
spe.perm<-adonis2(sitedata~strata,data=sites_to_strata,permutations=1000,method='euclidean')
spe.perm


#########
sites_to_strata = distinct(select(site_data, site, strata))
sites_to_strata = with(sites_to_strata, sites_to_strata[order(as.character(site)), ])
sites_to_strata = sites_to_strata[sites_to_strata$site %in% colnames(m), ]
# write.csv(sites_to_strata, 'classification/_output/sites_to_strata.csv', row.names = F)

## Community composition

# Get the top N most common species for each stratum (aggregated across all sites in that stratum)
N = 10
total_species_counts = data %>% group_by(common_name, strata, stage, thinned) %>% summarize(count = n()) %>% ungroup() %>% arrange(strata, desc(count))
top_N_species = total_species_counts %>% group_by(strata) %>% top_n(N, count) #%>% select(common_name)
(top_N_species = unique(top_N_species$common_name))

total_species_counts[total_species_counts$common_name=='Brown Creeper', ] # DEBUG
species_counts = total_species_counts[total_species_counts$common_name %in% top_N_species, ] # species subset only including the top N species from each strata

# Calculate relative abundance by strata
total_counts_by_strata = species_counts %>% group_by(strata) %>% summarize(total_count = sum(count))
top_N_species_by_strata = species_counts %>% left_join(total_counts_by_strata, by = "strata")
top_N_species_by_strata$rel_ab = top_N_species_by_strata$count / top_N_species_by_strata$total_count

# Plot
ggplot(top_N_species_by_strata, aes(fill=common_name, y=rel_ab, x=strata)) +
  geom_bar(position='stack', stat='identity') +
  scale_x_discrete(labels=strata_labels) +
  scale_fill_viridis_d(option='turbo') +
  labs(title = 'Acoustic community composition', x = 'Strata', y = 'Relative Vocalization Abundance', fill = 'Species') +
  theme_minimal()

# Calculate relative abundance by species
total_counts_by_species = species_counts %>% group_by(common_name) %>% summarize(total_count = sum(count))
top_N_species_by_species = species_counts %>% left_join(total_counts_by_species, by = 'common_name')
top_N_species_by_species$rel_ab = top_N_species_by_species$count / top_N_species_by_species$total_count

# Helper for names ordering
order = top_N_species_by_species %>% select(common_name, strata, rel_ab) %>% spread(strata, rel_ab) %>% group_by(common_name, `Stand Initiation`, `Competitive Exclusion`, Thinned, Mature) %>% replace(is.na(.), 0) %>% arrange(desc(Mature), `Stand Initiation`, Thinned)
# TODO: order by the majority occurance from Early to Mid to Mid (Thinned) to Late

ggplot(top_N_species_by_species, aes(fill=strata, y=rel_ab, x=factor(common_name, levels=order$common_name))) +
  geom_bar(position='stack', stat='identity') +
  scale_fill_manual(name = 'Strata', labels=strata_labels, values = strata_colors) +
  # scale_pattern_manual(name = 'Thinned', values = c('none', 'stripe')) +
  labs(title = '10 most common species detections per strata', x = '', y = 'Relative number of detections', fill = 'Strata') +
  coord_flip() +
  theme_minimal()

top_N_species_by_species_presab = top_N_species_by_species
top_N_species_by_species_presab$count = ifelse(top_N_species_by_species_presab$count > 1, 1, 0)

ggplot(top_N_species_by_species, aes(fill=strata, y=count, x=factor(common_name, levels=order$common_name))) +
  geom_bar(position='stack', stat='identity') +
  scale_fill_manual(name = 'Strata', labels=strata_labels, values = strata_colors) +
  # scale_pattern_manual(name = 'Thinned', values = c('none', 'stripe')) +
  labs(title = 'Acoustic community composition', x = 'Species (Common Name)', y = 'Relative Vocalization Abundance', fill = 'Strata') +
  coord_flip() +
  theme_minimal()

## Species richness

(species_richness_per_site = tapply(data$common_name, data$site, unique))
species_richness_per_site = species_richness_per_site[lengths(species_richness_per_site) > 0]
species_richness_per_site = lapply(species_richness_per_site, length)
species_richness_per_site = data.frame(t(data.frame(species_richness_per_site)))
colnames(species_richness_per_site) = c('count')
species_richness_per_site$site = rownames(species_richness_per_site)
species_richness_per_site = full_join(species_richness_per_site, site_data[,c('site', 'watershed', 'strata', 'stage', 'thinned')], by = c('site'))
species_richness_per_site = na.omit(species_richness_per_site)

ggplot(species_richness_per_site, aes(x=strata, y=count, fill=strata)) +
  # geom_boxplot_pattern(aes(pattern = thinned)) +
  geom_boxplot() + # NOTE: CUTTING OUTLIERS! outlier.shape = NA
  scale_x_discrete(labels=strata_labels) +
  # scale_fill_manual(name = 'Stage', values = stage_colors) +
  # scale_pattern_manual(name = 'Thinned', values = c('none', 'stripe')) +
  scale_fill_manual(name = 'Strata', values = strata_colors, labels=strata_labels) +
  labs(title = 'Species Richness', x = 'Strata', y = 'Acoustic Species Richness', fill = 'Species') +
  # ylim(5,25) +
  theme_minimal()

tapply(species_richness_per_site$count, species_richness_per_site$strata, mean)
# Hagar et al 1996 found 24.5 thinned, 22.6 unthinned
# Jones et al 2012 found 20-60 in early seral

# Richness over time
richness_by_date_and_strata = data %>%
  group_by(site, date, strata) %>%
  summarize(richness = n_distinct(common_name))

rrr = richness_by_date_and_strata %>%
  group_by(date, strata) %>%
  summarize(avg_richness = median(richness))

# standinit = richness_by_date_and_strata[richness_by_date_and_strata$strata=='Stand Initiation',]
# spline_int = as.data.frame(spline(standinit$date, standinit$total_species))
# ggplot() + geom_line(spline_int, mapping=aes(x=x, y=y))

temp = read.csv('classification/_output/temp_2021.csv')
temp = temp[c('datetime', 'tempmax', 'temp')]
temp$date = as.Date(temp$datetime)

richness_by_date_and_strata = full_join(richness_by_date_and_strata, temp, by=c('date'))
richness_by_date_and_strata = na.omit(richness_by_date_and_strata)

ggplot(richness_by_date_and_strata, aes(x = date, y = richness, color = strata)) +
  geom_smooth(method = 'loess', span = 0.2, se = F) +
  geom_line(mapping=aes(x = date, y = tempmax), color='#FF000055') +
  geom_point(shape=16, alpha=0.25) +
  scale_color_manual(name = 'Strata', values = strata_colors, labels=strata_labels) +
  labs(title = 'Species richness over time', x = 'Date', y = 'Species Richness') +
  annotate("rect",xmin=as.Date('2021-06-26'),xmax=as.Date('2021-06-29'),ymin=-Inf,ymax=Inf, alpha=0.1, fill='red') + theme(text = element_text(size = 18))

#####

sitedatedata <- data %>%
  group_by(site, date, common_name) %>%
  summarize(count = n())
sitedatedata$sitedate = paste0(sitedatedata$site, '_', sitedatedata$date)
sitedatedata = full_join(sitedatedata, site_data[,c('date', 'site', 'strata')], by = c('site', 'date'), relationship = "many-to-many")
sitedatedata = sitedatedata[,c('sitedate', 'common_name', 'count')]
sitedatedata$sitedate = factor(sitedatedata$sitedate)
sitedatedata = as.data.frame(sitedatedata)

df <- create.matrix(sitedatedata, tax.name = "common_name",
                    locality = "sitedate",
                    abund.col = "count",
                    abund = TRUE)
sitedatebirddata <- as.data.frame(t(df))
sitedatebirddata <- drop.var(sitedatebirddata,min.po=1)
sitedatebirddata<-decostand(sitedatebirddata, method='hel')
bird.sitedate.pca<-PCA(sitedatebirddata,graph=F)

# Plot like the others, but now each point is a site-date combination
fviz_screeplot(bird.sitedate.pca, addlabels = TRUE, ylim = c(0, 30))
fviz_pca_var(bird.sitedate.pca, col.var="contrib", gradient.cols = gradient.cols, repel = F, title="PCA variables (species)", ggtheme = theme(text = element_text(size = 18)))
fviz_pca_biplot(bird.sitedate.pca, label="var",col.var="contrib",repel = F, title="", ggtheme = theme(text = element_text(size = 18)))
####
site_date_PCAscores = as.data.frame((bird.sitedate.pca$ind)$coord)
site_date_PCAscores$date = as.Date(sapply(strsplit(rownames(site_date_PCAscores), '_'), '[[', 2))
site_date_PCAscores$site = sapply(strsplit(rownames(site_date_PCAscores), '_'), '[[', 1)
site_date_PCAscores = full_join(site_date_PCAscores, sites_to_strata[,c('site', 'strata')], by = c('site'))
site_date_PCAscores = full_join(site_date_PCAscores, temp, by=c('date'))
site_date_PCAscores = na.omit(site_date_PCAscores)

ggplot(site_date_PCAscores) +
  geom_smooth(method = 'loess', span = 0.15, se = F, mapping=aes(x = date, y = Dim.1, color = strata)) +
  #geom_line(mapping=aes(x = date, y = Dim.2, color = strata)) +
  geom_point(shape=16, alpha=0.5, mapping=aes(x = date, y = Dim.1, color = strata)) +
  scale_color_manual(name = 'Strata', values = strata_colors) +
  geom_line(mapping=aes(x = date, y = tempmax-15), color='#FF000055') +
  labs(x = 'Date', y = 'Dim 2') +
  annotate("rect",xmin=as.Date('2021-06-26'),xmax=as.Date('2021-06-29'),ymin=-Inf,ymax=Inf, alpha=0.1, fill='red') + theme(text = element_text(size = 18))


ggplot(site_date_PCAscores) +
  geom_smooth(method = 'loess', span = 0.15, se = F, mapping=aes(x = date, y = Dim.2, color = strata)) +
  #geom_line(mapping=aes(x = date, y = Dim.2, color = strata)) +
  geom_point(shape=16, alpha=0.5, mapping=aes(x = date, y = Dim.2, color = strata)) +
  scale_color_manual(name = 'Strata', values = strata_colors) +
  geom_line(mapping=aes(x = date, y = tempmax-15), color='#FF000055') +
  labs(x = 'Date', y = 'Dim 2') +
  annotate("rect",xmin=as.Date('2021-06-26'),xmax=as.Date('2021-06-29'),ymin=-Inf,ymax=Inf, alpha=0.1, fill='red') + theme(text = element_text(size = 18))

ggplot(temp, aes(x = date, y = tempmax)) +
  # geom_smooth(method = 'loess', span = 0.1, se = F) +
  geom_line(temp, aes(x = date, y = tempmax)) +
  geom_point(shape=16, alpha=0.35) +
  scale_color_manual(name = 'Strata', values = strata_colors) +
  labs(x = 'Date', y = 'Dim 2') +
  annotate("rect",xmin=as.Date('2021-06-26'),xmax=as.Date('2021-06-29'),ymin=-Inf,ymax=Inf, alpha=0.1, fill='red') + theme(text = element_text(size = 18))



#####
# ACI
library(readr)
results <- read_csv("acoustic_indices/_output/2021/results.csv")
results = results %>%
  rename(serial_no = SerialNo, date = SurveyDate) %>% select(ACI, serial_no, date, TimeStart)
results = full_join(results, site_data[,c('serial_no', 'date', 'deploy', 'site', 'watershed', 'strata', 'stage', 'thinned')], by = c('serial_no', 'date'))
results$serial_no = factor(results$serial_no)
results = na.omit(results)

meanaci = results %>% group_by(date, serial_no) %>% summarize(meanACI = mean(ACI))
meanaci = full_join(meanaci, site_data[,c('serial_no', 'date','strata')], by = c('serial_no', 'date'))
meanaci = full_join(meanaci, temp, by=c('date'))

library("scales")
ggplot(meanaci) +
  geom_smooth(method = 'loess', span = 0.1, se = F, mapping=aes(x = date, y = meanACI, color = strata)) +
  #geom_line(mapping=aes(x = date, y = Dim.2, color = strata)) +
  geom_point(shape=16, alpha=0.5, mapping=aes(x = date, y = meanACI, color = strata)) +
  scale_color_manual(name = 'Strata', values = strata_colors) +
  geom_line(mapping=aes(x = date, y = tempmax*10+800), color='#FF000055') +
  labs(x = 'Date', y = 'ACI', title = 'Acoustic complexity index') +
  annotate("rect",xmin=as.Date('2021-06-26'),xmax=as.Date('2021-06-29'),ymin=-Inf,ymax=Inf, alpha=0.1, fill='red') + theme(text = element_text(size = 18)) + scale_x_date(limits = as.Date(c("2021-04-14", "2021-08-01")))

