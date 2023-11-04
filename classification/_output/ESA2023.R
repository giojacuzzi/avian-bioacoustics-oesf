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
library(ggpubr)
source("~/Dropbox/Research/ESA_Julian_Analysis/biostats.R", encoding = 'UTF-8')

stage_colors  = c('#73E2A7', '#1C7C54', '#6c584c')
strata_colors = c('#73E2A7', '#1C7C54',  '#9200AC', '#6c584c')
prescription_colors = c('#73E2A7', '#3668FA', '#1C7C54', '#9200AC', '#6c584c')
gradient.cols = gradient.cols = c("#00AFBB22", "#E7B800", "#FC4E07")
strata_labels = c('Early', 'Mid', 'Mid (Thinned)', 'Late')
prescription_labels = c('Standard Regen.', 'Complex Early', 'Not Thinned', 'Thinned', 'Mature')
prescription_order = c('Stand Initiation', 'Complex Early', 'Competitive Exclusion', 'Thinned', 'Mature')

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
data$prescription = as.character(data$strata)
data[(data$watershed %in% c('Az', 'Bz', 'Cz', 'Dz') & data$prescription == 'Stand Initiation'), 'prescription'] = 'Complex Early'
data$prescription = factor(data$prescription)

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
data$common_name = factor(data$common_name) # reduce factor levels

## Export total unique detections for multivariate ordination analysis
sort(as.character(unique(data$date)))
observations_per_date_site = data %>% select(date, site, strata, common_name) %>% group_by(date, site, strata, common_name) %>% summarise(count = n())
factors = c('site', 'strata', 'common_name')
observations_per_date_site[factors] = lapply(observations_per_date_site[factors], factor)
observations_per_date_site = as.data.frame(observations_per_date_site)

# # Individual species detections over time
# # TODO: mark zeros for days on which we do have data
# # Are there periods of no activity in the weeks following the heat wave event?
# dds = sort(unique(data[data$strata=='Stand Initiation', 'date']))
# 
# library(scales)
# theme_set(theme_minimal())
# hotties = observations_per_date_site[observations_per_date_site$strata=='Stand Initiation',]
# hotties = tidyr::complete(hotties, date = seq(min(date), max(date), by = "day"))
# for (species in levels(hotties$common_name)) {
#   p = ggplot(hotties[hotties$common_name==species,], aes(x = date, y = count)) + geom_point() + scale_x_date(limits = as.Date(c("2021-04-14", "2021-08-01"))) + labs(title=species) + annotate("rect",xmin=as.Date('2021-06-26'),xmax=as.Date('2021-06-29'),ymin=-Inf,ymax=Inf, alpha=0.1, fill='red') + annotate("rect",xmin=as.Date('2021-06-01'),xmax=as.Date('2021-06-02'),ymin=-Inf,ymax=Inf, alpha=0.1, fill='red') + annotate("rect",xmin=as.Date('2021-07-05'),xmax=as.Date('2021-07-21'),ymin=-Inf,ymax=Inf, alpha=0.1, fill='blue')
#   print(p)
#   readline('Next?')
# }

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

# Complex Early Seral Tag
sites_to_strata$prescription = as.character(sites_to_strata$strata)
sites_to_strata[(grepl('z', as.character(sites_to_strata$site), fixed = T) & sites_to_strata$prescription == 'Stand Initiation'), 'prescription'] = 'Complex Early'
sites_to_strata$prescription = factor(sites_to_strata$prescription)
sss = as.character(unique(sites_to_strata$site))
sites_to_strata = full_join(sites_to_strata, site_data[,c('site', 'stage')], by = c('site'))
sites_to_strata = distinct(select(sites_to_strata, site, strata, stage, prescription))
sites_to_strata = sites_to_strata[sites_to_strata$site %in% sss, ]
sites_to_strata$prescription = factor(sites_to_strata$prescription, levels=prescription_order)

###### PCA
sitedata = t(mat)
sitedata <- drop.var(sitedata,min.po=1)

# Set to presence/absence 1 or 0
# sitedata[-1] <- as.integer(sitedata[-1] > 1) # TODO: relab or p/a?
sitedata<-decostand(sitedata, method='hel')
bird.pca<-PCA(sitedata,graph=F)

theme_set(theme_minimal())

fviz_screeplot(bird.pca, addlabels = TRUE, ylim = c(0, 30))
pca_loadings = fviz_pca_var(bird.pca, col.var="contrib", gradient.cols = gradient.cols, repel = F, title="", ggtheme = theme(text = element_text(size = 18))); pca_loadings

ggsave(pca_loadings + theme(legend.position="none", text=element_text(size=22), plot.margin = margin(1,1,1,1, 'cm')), file='classification/_figures/pca_loadings.png', width=10, height=9)

fviz_pca_biplot(bird.pca, label="var",col.var="contrib",repel = F, title="", ggtheme = theme(text = element_text(size = 18)))

# Stage
pca_stage = fviz_pca_ind(bird.pca, geom = "point", habillage = sites_to_strata$stage, palette = stage_colors, 
             addEllipses = TRUE, ellipse.level=0.95,
             title="", ggtheme = theme(text = element_text(size = 18))) + xlim(-5, 15) + ylim(-15, 15); pca_stage

adonis2(sitedata~stage,data=sites_to_strata,permutations=1000,method='euclidean')

ggsave(pca_stage + theme(legend.position="none", text=element_text(size=22), plot.margin = margin(1,1,1,1, 'cm')), file='classification/_figures/pca_stage.png', width=10, height=9)

ggsave(as_ggplot(get_legend(pca_stage)), file='classification/_figures/pca_stage_legend.png', width=8, height=8)

# Prescription
sites_to_strata$prescription = factor(sites_to_strata$prescription, levels=c('Stand Initiation', 'Competitive Exclusion', 'Mature', 'Complex Early', 'Thinned'))
pca_prescription = fviz_pca_ind(bird.pca, geom = "point", habillage = sites_to_strata$prescription,palette = c('#73E2A7', '#1C7C54', '#6c584c', '#3668FA', '#9200AC'),
             addEllipses = TRUE, ellipse.level=0.95,
             title="", ggtheme = theme(text = element_text(size = 18)))+ xlim(-5, 15) + ylim(-15, 15); pca_prescription

ggsave(pca_prescription + theme(legend.position="none", text=element_text(size=22), plot.margin = margin(1,1,1,1, 'cm')), file='classification/_figures/pca_prescription.png', width=10, height=9)

ggsave(as_ggplot(get_legend(pca_prescription)), file='classification/_figures/pca_prescription_legend.png', width=8, height=8)

#########
sites_to_strata = distinct(select(site_data, site, strata))
sites_to_strata = with(sites_to_strata, sites_to_strata[order(as.character(site)), ])
sites_to_strata = sites_to_strata[sites_to_strata$site %in% colnames(m), ]
# write.csv(sites_to_strata, 'classification/_output/sites_to_strata.csv', row.names = F)

## Community composition

# Get the top N most common species for each stratum (aggregated across all sites in that stratum)
N = 10
total_species_counts = data %>% group_by(common_name, stage) %>% summarize(count = n()) %>% ungroup() %>% arrange(stage, desc(count))
top_N_species = total_species_counts %>% group_by(stage) %>% top_n(N, count) #%>% select(common_name)
(top_N_species = unique(top_N_species$common_name))
species_counts = total_species_counts[total_species_counts$common_name %in% top_N_species, ] # species subset only including the top N species from each strata

# Calculate relative abundance by strata
total_counts_by_strata = species_counts %>% group_by(stage) %>% summarize(total_count = sum(count))
top_N_species_by_strata = species_counts %>% left_join(total_counts_by_strata, by = "stage")
top_N_species_by_strata$rel_ab = top_N_species_by_strata$count / top_N_species_by_strata$total_count

# Plot
ggplot(top_N_species_by_strata, aes(fill=common_name, y=rel_ab, x=stage)) +
  geom_bar(position='stack', stat='identity') +
  # scale_x_discrete(labels=strata_labels) +
  scale_fill_viridis_d(option='turbo') +
  labs(title = 'Acoustic community composition', x = 'Stage', y = 'Relative Vocalization Abundance', fill = 'Species') +
  theme_minimal()

# Calculate relative abundance by species
total_counts_by_species = species_counts %>% group_by(common_name) %>% summarize(total_count = sum(count))
top_N_species_by_species = species_counts %>% left_join(total_counts_by_species, by = 'common_name')
top_N_species_by_species$rel_ab = top_N_species_by_species$count / top_N_species_by_species$total_count

# Helper for names ordering
order = top_N_species_by_species %>% select(common_name, stage, rel_ab) %>% spread(stage, rel_ab) %>% group_by(common_name, `Early`, `Mid`, `Late`) %>% replace(is.na(.), 0) %>% arrange(desc(Late), `Early`, Mid)
# TODO: order by the majority occurance from Early to Mid to Mid (Thinned) to Late

# 10 most common species detections per strata
p_topN_stage = ggplot(top_N_species_by_species, aes(fill=stage, y=rel_ab, x=factor(common_name, levels=order$common_name))) +
  geom_bar(position='stack', stat='identity') +
  scale_fill_manual(name = 'Strata', labels=strata_labels, values = stage_colors) +
  labs(title = '', x = '', y = 'Proportion of detections', fill = 'Strata') +
  coord_flip() +
  theme_minimal(); p_topN_stage
ggsave(p_topN_stage + theme(legend.position="none", text=element_text(size=22), plot.margin = margin(1,1,1,1, 'cm')), file='classification/_figures/topN_stage.png', width=8, height=10)

ggsave(as_ggplot(get_legend(p_topN_stage)), file='classification/_figures/topN_stage_legend.png', width=8, height=8)

##### PRESCRIPTION ######################

total_species_counts = data %>% group_by(common_name, prescription) %>% summarize(count = n()) %>% ungroup() %>% arrange(prescription, desc(count))
species_counts = total_species_counts[total_species_counts$common_name %in% top_N_species, ] # only keep top N from before

# Calculate relative abundance by strata
total_counts_by_strata = species_counts %>% group_by(prescription) %>% summarize(total_count = sum(count))
top_N_species_by_strata = species_counts %>% left_join(total_counts_by_strata, by = "prescription")
top_N_species_by_strata$rel_ab = top_N_species_by_strata$count / top_N_species_by_strata$total_count

# Calculate relative abundance by species
total_counts_by_species = species_counts %>% group_by(common_name) %>% summarize(total_count = sum(count))
top_N_species_by_species = species_counts %>% left_join(total_counts_by_species, by = 'common_name')
top_N_species_by_species$rel_ab = top_N_species_by_species$count / top_N_species_by_species$total_count

# 10 most common species detections per strata
# top_N_species_by_species$prescription = factor(top_N_species_by_species$prescription, labels = c('Stand Initiation', 'Complex Early', 'Competitive Exclusion', 'Thinned', 'Mature'))
p_topN_stage = ggplot(top_N_species_by_species, aes(fill=prescription, y=rel_ab, x=factor(common_name, levels=order$common_name))) +
  geom_bar(position='stack', stat='identity') +
  labs(title = '', x = '', y = 'Proportion of detections') +
  coord_flip() +
  # scale_fill_discrete(breaks=c('Stand Initiation', 'Complex Early', 'Competitive Exclusion', 'Thinned', 'Mature')) +
  scale_fill_manual(name= 'Prescription', breaks=c('Stand Initiation', 'Competitive Exclusion', 'Mature', 'Complex Early', 'Thinned'), values = c("#73E2A7", "#1C7C54", "#6c584c", "#3668FA", "#9200AC"), labels = c('Standard Regen.', 'Not Thinned', 'Mature', 'Complex Early', 'Thinned')) +
  theme_minimal(); p_topN_stage
ggsave(p_topN_stage + theme(legend.position="none", text=element_text(size=22), plot.margin = margin(1,1,1,1, 'cm')), file='classification/_figures/topN_prescription.png', width=8, height=10)

ggsave(as_ggplot(get_legend(p_topN_stage)), file='classification/_figures/topN_prescription_legend.png', width=8, height=8)

# p_topN_prescription = ggplot(top_N_species_by_species, aes(fill=prescription, y=rel_ab, x=factor(common_name, levels=order$common_name))) +
#   geom_bar(position='stack', stat='identity') +
#   scale_fill_manual(name = 'Strata', labels=strata_labels, values = stage_colors) +
#   labs(title = '', x = '', y = 'Proportion of detections', fill = 'Strata') +
#   coord_flip() +
#   theme_minimal(); p_topN_prescription

top_N_species_by_species_presab = top_N_species_by_species
top_N_species_by_species_presab$count = ifelse(top_N_species_by_species_presab$count > 1, 1, 0)

# ggplot(top_N_species_by_species, aes(fill=strata, y=count, x=factor(common_name, levels=order$common_name))) +
#   geom_bar(position='stack', stat='identity') +
#   scale_fill_manual(name = 'Strata', labels=strata_labels, values = strata_colors) +
#   # scale_pattern_manual(name = 'Thinned', values = c('none', 'stripe')) +
#   labs(title = 'Acoustic community composition', x = 'Species (Common Name)', y = 'Relative Vocalization Abundance', fill = 'Strata') +
#   coord_flip() +
#   theme_minimal()

## Species richness
site_data$prescription = as.character(site_data$strata)
site_data[(grepl('z', as.character(site_data$site), fixed = T) & site_data$prescription == 'Stand Initiation'), 'prescription'] = 'Complex Early'
site_data$prescription = factor(site_data$prescription)

(species_richness_per_site = tapply(data$common_name, data$site, unique))
species_richness_per_site = species_richness_per_site[lengths(species_richness_per_site) > 0]
species_richness_per_site = lapply(species_richness_per_site, length)
species_richness_per_site = data.frame(t(data.frame(species_richness_per_site)))
colnames(species_richness_per_site) = c('count')
species_richness_per_site$site = rownames(species_richness_per_site)
species_richness_per_site = full_join(species_richness_per_site, site_data[,c('site', 'watershed', 'strata', 'stage', 'prescription')], by = c('site'))
species_richness_per_site = na.omit(species_richness_per_site)
species_richness_per_site$prescription = factor(species_richness_per_site$prescription, levels=prescription_order)

p_sr_site = ggplot(species_richness_per_site, aes(x=stage, y=count, fill=stage)) +
  geom_boxplot() + # NOTE: CUTTING OUTLIERS! outlier.shape = NA
  # scale_x_discrete(labels=) +
  # scale_fill_manual(name = 'Stage', values = stage_colors) +
  # scale_pattern_manual(name = 'Thinned', values = c('none', 'stripe')) +
  scale_fill_manual(name = 'Stage', values = stage_colors) +
  labs(title = 'Species Richness', x = 'Stage', y = 'Acoustic Species Richness') +
  ylim(10,50) +
  theme_minimal(); p_sr_site
ggsave(p_sr_site + theme(legend.position="none", text=element_text(size=22), plot.margin = margin(1,1,1,1, 'cm')), file='classification/_figures/sr_site.png', width=8, height=10)

ggsave(as_ggplot(get_legend(p_sr_site)), file='classification/_figures/sr_site_legend.png', width=8, height=8)


p_sr_prescription = ggplot(species_richness_per_site, aes(x=stage, y=count, fill=prescription)) +
  # geom_boxplot_pattern(aes(pattern = thinned)) +
  geom_boxplot() + # NOTE: CUTTING OUTLIERS! outlier.shape = NA
  # scale_x_discrete(labels=strata_labels) +
  # scale_fill_manual(name = 'Stage', values = stage_colors) +
  # scale_pattern_manual(name = 'Thinned', values = c('none', 'stripe')) +
  scale_fill_manual(name = 'Prescription', values = prescription_colors, labels = prescription_labels) +
  labs(title = 'Species Richness', x = 'Strata', y = 'Acoustic Species Richness') +
  ylim(10,50) +
  theme_minimal(); p_sr_prescription
ggsave(p_sr_prescription + theme(legend.position="none", text=element_text(size=22), plot.margin = margin(1,1,1,1, 'cm')), file='classification/_figures/sr_prescription.png', width=8, height=10)

ggsave(as_ggplot(get_legend(p_sr_prescription)), file='classification/_figures/sr_prescription_legend.png', width=8, height=8)

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

pt = ggplot(temp, aes(x = date, y = tempmax)) +
  # geom_smooth(method = 'loess', span = 0.1, se = F) +
  geom_line(temp, mapping=aes(x = date, y = tempmax)) +
  # geom_point(shape=16, alpha=0.35) +
  scale_color_manual(name = 'Strata', values = strata_colors) +
  labs(x = 'Date', y = 'Temperature (C)') +
  annotate("rect",xmin=as.Date('2021-06-25'),xmax=as.Date('2021-06-29'),ymin=-Inf,ymax=Inf, alpha=0.1, fill='red') + theme(text = element_text(size = 18)); pt
ggsave(pt, file='classification/_figures/temp.png', width=8, height=8)


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

