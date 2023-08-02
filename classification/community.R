source('global.R')

## Preprocess data

out_dir = 'classification/_output'

file_list <- list.files(path = out_dir, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
combined_data <- do.call(rbind, lapply(file_list, read.csv, colClasses = c(
  'common_name' = 'factor',
  'scientific_name' = 'factor',
  'start_date' = 'POSIXct',
  'end_date' = 'POSIXct',
  'serial_no' = 'factor',
  'file' = 'factor'
)))

# Only consider detections with a high enough confidence
min_confidence = 0.99
data = combined_data[combined_data$confidence>=min_confidence,]
data$date = as.Date(data$start_date)

site_data = get_site_strata_date_serial_data()
data = full_join(data, site_data[,c('serial_no', 'date', 'site', 'watershed', 'strata', 'stage', 'thinned')], by = c('serial_no', 'date'))
data = na.omit(data)

## Explore data

# (unique_species = tapply(data$common_name, data$strata, unique))
# summary(unique_species)
# 
# summary(data[data$strata=='Stand Initiation', 'common_name'])
# summary(data[data$strata=='Competitive Exclusion', 'common_name'])
# summary(data[data$strata=='Thinned', 'common_name'])
# summary(data[data$strata=='Mature', 'common_name'])
# 
# indicators = c(
#   'Brown Creeper',
#   'Bewick\'s Wren',
#   'Chestnut-backed Chickadee',
#   'Hutton\'s Vireo',
#   'Orange-crowned Warbler',
#   'Pacific Wren',
#   'Pacific-slope Flycatcher',
#   'Pileated Woodpecker',
#   'Varied Thrush',
#   'Wilson\'s Warbler'
# )
# 
# data_indicators = data[data$common_name %in% indicators,]
# sort(as.character(unique(data_indicators$common_name)))
# 
# (unique_indicators = tapply(data_indicators$common_name, data_indicators$strata, unique))
# summary(unique_indicators)

## Community composition

# Get the top N most common species across all strata
N = 10
factor_counts = table(data$common_name)
sorted_factor_levels = names(factor_counts[order(-factor_counts)])
top_N_factor_levels = head(sorted_factor_levels, N)

# Filter the dataframe to include only rows with the top N species
# TODO: top N species per strata?
data_top = data[data$common_name %in% top_N_factor_levels, ]
data_top$common_name = factor(data_top$common_name)
top_relative_abundance = tapply(data_top$common_name, data_top$strata, table)

library(dplyr)
library(tidyr)
df_top_rel_ab = top_relative_abundance %>% lapply(., function(i) spread(as.data.frame(i), Var1, Freq)) %>% bind_rows() %>% t() %>% as.data.frame()
colnames(df_top_rel_ab) = names(top_relative_abundance)
df_top_rel_ab = as.data.frame(apply(df_top_rel_ab, 2, function(x) { return(100 * x / sum(x)) })) # convert to percentages
df_top_rel_ab$species = rownames(df_top_rel_ab)
df_top_rel_ab$species = factor(df_top_rel_ab$species)

df_top_rel_ab_long <- gather(df_top_rel_ab, strata, count, 1:4, factor_key=T) # convert to long format for plotting

library(ggplot2)
library(viridis)
ggplot(df_top_rel_ab_long, aes(fill=species, y=count, x=strata)) + geom_bar(position='stack', stat='identity') + labs(title = 'Acoustic Community Composition', x = 'Strata', y = 'Relative Vocalization Abundance', fill = 'Species') + scale_fill_viridis_d() + theme_minimal()

## More composition

community = tapply(data$common_name, data$strata, table) %>% lapply(., function(i) spread(as.data.frame(i), Var1, Freq)) %>% bind_rows() %>% t() %>% as.data.frame()
community = community[rowSums(community[])>0,]
colnames(community) = names(tapply(data$common_name, data$strata, table))
community$species = rownames(community)

community_long <- gather(community, strata, count, 1:4, factor_key=T) # convert to long format for plotting

ggplot(community_long, aes(strata, species)) +
  geom_tile(aes(fill = count)) +
  geom_text(aes(label = count)) +
  scale_fill_gradient(low = "white", high = "red") + theme_minimal()

## Richness

(species_richness_per_site = tapply(data$common_name, data$site, unique))
species_richness_per_site = species_richness_per_site[lengths(species_richness_per_site) > 0]
species_richness_per_site = lapply(species_richness_per_site, length)
species_richness_per_site = data.frame(t(data.frame(species_richness_per_site)))
colnames(species_richness_per_site) = c('count')
species_richness_per_site$site = rownames(species_richness_per_site)
species_richness_per_site = full_join(species_richness_per_site, site_data[,c('site', 'watershed', 'strata', 'stage', 'thinned')], by = c('site'))
species_richness_per_site = na.omit(species_richness_per_site)

ggplot(species_richness_per_site, aes(x=strata, y=count)) + geom_boxplot() + labs(title = 'Species Richness', x = 'Strata', y = 'Acoustic Species Richness', fill = 'Species') + theme_minimal()
