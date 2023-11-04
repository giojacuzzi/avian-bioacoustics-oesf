source('global.R')
library(dplyr)
library(tidyr)
library(ggplot2)
library(viridis)
library(ggpattern)

stage_colors  = c('#73E2A7', '#1C7C54', '#6c584c')
strata_colors = c('#73E2A7', '#1C7C54',  '#73125A', '#6c584c')
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

# Only consider detections with a high enough confidence
min_confidence = 0.95
data = combined_data[combined_data$confidence>=min_confidence,]

# NOTE: original dates are incorrect!
data$date = get_date_from_file_name(as.character(data[,'file'])) #as.Date(data$start_date)

site_data = get_site_strata_date_serial_data()

# Complex early seral tag
site_data$comp_early_seral = FALSE
site_data[site_data$watershed %in% c('Az', 'Bz', 'Cz', 'Dz') & site_data$strata == 'Stand Initiation', 'comp_early_seral'] = TRUE

data = full_join(data, site_data[,c('serial_no', 'date', 'site', 'watershed', 'strata', 'stage', 'thinned')], by = c('serial_no', 'date'))
data = na.omit(data)
data$common_name = factor(data$common_name) # reduce factor levels

# Remove species that were only observed on a single day
# TODO: # Group by scientific_name and count unique dates
# # Filter birds that were observed on only one date
# bird_counts <- data %>%
#   group_by(common_name) %>%
#   summarise(num_dates = n_distinct(date))
# birds_observed_once <- bird_counts %>% filter(num_dates == 1)
sort(summary(data$common_name), decreasing=F)[1:10]
more_than_once = c( # TODO: do this right!
  "American Robin","Belted Kingfisher","Black-headed Grosbeak","Black-throated Gray Warbler","Brown Creeper","Cedar Waxwing","Chestnut-backed Chickadee","Common Raven","Dark-eyed Junco","Evening Grosbeak","Golden-crowned Kinglet","Hairy Woodpecker","Hammond's Flycatcher","Marbled Murrelet","Mountain Chickadee","Olive-sided Flycatcher","Pacific Wren","Pacific-slope Flycatcher","Pileated Woodpecker","Pine Siskin","Red-breasted Nuthatch","Steller's Jay","Townsend's Warbler","Varied Thrush","Vaux's Swift","Western Tanager","Band-tailed Pigeon","Hermit Thrush","Northern Flicker","Rufous Hummingbird","Spotted Towhee","Wilson's Warbler","Hutton's Vireo","MacGillivray's Warbler","Orange-crowned Warbler","Purple Finch","Swainson's Thrush","Violet-green Swallow","Sooty Grouse","Warbling Vireo","Sandhill Crane","White-crowned Sparrow","Bald Eagle","Green-winged Teal","Red-tailed Hawk","Song Sparrow","Willow Flycatcher","Greater White-fronted Goose","Ruby-crowned Kinglet","Common Nighthawk","Western Bluebird"
)
data = data[data$common_name %in% more_than_once, ]
data$common_name = factor(data$common_name) # reduce factor levels

## Export total unique detections for multivariate ordination analysis
sort(as.character(unique(data$date)))
observations_per_date_site = data %>% select(date, site, strata, common_name) %>% group_by(date, site, strata, common_name) %>% summarise(count = n())
factors = c('site', 'strata', 'common_name')
observations_per_date_site[factors] = lapply(observations_per_date_site[factors], factor)
observations_per_date_site = as.data.frame(observations_per_date_site)

write.csv(observations_per_date_site, 'classification/_output/_birdcounts_oesf.csv', row.names = F)

# Create full matrix of all species per site (aggregating dates), including nondetections (count == 0)
library(fossil)
m = create.matrix(observations_per_date_site, tax.name = 'common_name', locality = 'site', abund.col = 'count', abund = T)
n_dates_per_site = as.data.frame(tapply(observations_per_date_site$date, observations_per_date_site$site, function(x) { length(unique(x))}))
n_dates_per_site$site = rownames(n_dates_per_site)
colnames(n_dates_per_site) = c('n_dates', 'site')

test = m
df_divisor = t(n_dates_per_site['n_dates'])
for (col_name in colnames(test)) {
  print(col_name)
  if (col_name %in% colnames(df_divisor)) {
    test[, col_name] = test[, col_name] / df_divisor['n_dates', col_name]
  }
}
mat = apply(test, c(1, 2), round) # round
# mat <- apply(test, c(1, 2), function(x) { ifelse(x < 1, 0, round(x)) }) # remove less than one average values (assumed transient detections), and round

write.csv(mat, 'classification/_output/birdcount_average_per_site_oesf.csv', row.names = T)

sites_to_strata = distinct(select(site_data, site, strata, comp_early_seral))
sites_to_strata = with(sites_to_strata, sites_to_strata[order(as.character(site)), ])
sites_to_strata = sites_to_strata[sites_to_strata$site %in% colnames(mat), ]
write.csv(sites_to_strata, 'classification/_output/sites_to_strata.csv', row.names = F)

# # Create a full table of all species per site and date
# df = as.data.frame(observations_per_date_site)
# df$common_name = factor(df$common_name)
# df = select(df, -strata)
# test = complete(df, date, site, common_name, fill=list(count=0))
# 
# # Average by date per site to flatten the date dimension
# averaged_counts_per_site = test %>% group_by(site, date, common_name) %>% summarise(average_count = mean(count))
# 
# ##

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
  labs(title = 'Acoustic community composition', x = 'Species (Common Name)', y = 'Relative Vocalization Abundance', fill = 'Strata') +
  coord_flip() +
  theme_minimal()

# Associations
# TODO: also arrange by count per strata
head(order %>% arrange(desc(Mature)))
head(order %>% arrange(desc(Thinned)))
head(order %>% arrange(desc(`Competitive Exclusion`)))
head(order %>% arrange(desc(`Stand Initiation`)))

## Detailed community composition

community = tapply(data$common_name, data$strata, table) %>% lapply(., function(i) spread(as.data.frame(i), Var1, Freq)) %>% bind_rows() %>% t() %>% as.data.frame()
community = community[rowSums(community[])>0,]
colnames(community) = names(tapply(data$common_name, data$strata, table))
community$species = rownames(community)

community_long <- gather(community, strata, count, 1:4, factor_key=T) # convert to long format for plotting

ggplot(community_long, aes(strata, species)) +
  geom_tile(aes(fill = count)) +
  geom_text(aes(label = count)) +
  scale_fill_gradient(low = "white", high = "red") + theme_minimal()

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

# Richness over time

# standinit = richness_by_date_and_strata[richness_by_date_and_strata$strata=='Stand Initiation',]
# spline_int = as.data.frame(spline(standinit$date, standinit$total_species))
# ggplot() + geom_line(spline_int, mapping=aes(x=x, y=y))

ggplot(richness_by_date_and_strata, aes(x = date, y = total_species, color = strata)) +
  geom_smooth(method = 'loess', span = 0.5, se = F) +
  # geom_line() +
  # geom_point(shape=16, alpha=0.15) +
  scale_color_manual(name = 'Strata', values = strata_colors, labels=strata_labels) +
  labs(title = 'Species richness over time', x = 'Date', y = 'Species Richness') +
  annotate("rect",xmin=as.Date('2021-06-26'),xmax=as.Date('2021-06-29'),ymin=-Inf,ymax=Inf, alpha=0.1, fill='red') + theme(text = element_text(size = 18))

rich_during_heatwave = richness_by_date_and_strata[richness_by_date_and_strata$date %in% c('2021-06-26', '2021-06-27', '2021-06-28'),]
tapply(rich_during_heatwave$total_species, rich_during_heatwave$strata, mean)
tapply(species_richness_per_site$count, species_richness_per_site$strata, mean)

# Fill missing dates with NA
full = richness_by_date_and_strata %>% group_by(strata) %>% complete(date = seq(min(date), max(date), by = 'day'))

ggplot(full %>% group_by(strata) %>% fill(total_species, .direction = "downup"), aes(x = date, y = total_species, color = strata)) +
  geom_smooth(method = 'loess', span = 0.31, se = F) +
  # geom_line() +
  # geom_point() +
  scale_color_manual(name = 'Strata', values = strata_colors) +
  labs(title = 'Species richness over time', x = 'Date', y = 'Species Richness') +
  annotate("rect",xmin=as.Date('2021-06-26'),xmax=as.Date('2021-06-29'),ymin=-Inf,ymax=Inf, alpha=0.1, fill='red') +
  theme_minimal()
