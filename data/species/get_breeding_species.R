###
library(tidyverse)

## Johnson and O'Neil 2001
species_JO = read.csv('data/species/species_birds_WA_JohnsonOneil2001.csv')
species_JO = species_JO[(species_JO$Occurrence %in% c('Occurs', 'Non-native')) & species_JO$Breeding_Status=='Breeds', ]
nrow(species_JO)

## eBird via BirdNET-Analyzer
# To get potential species from eBird, run:
# cd repos/BirdNET-Analyzer
# python3 species.py --o '../olympic-songbirds/data/species/species_eBird.txt' --lat 47.63610 --lon -124.37216 --week -1

species_eBird = strsplit(read_lines('data/species/species_eBird.txt'), '_')
species_eBird = data.frame(
  Name_Scientific = sapply(species_eBird, `[`, 1),
  Name_Common = sapply(species_eBird, `[`, 2)
)
nrow(species_eBird)

# Inspect merge. Note that JO appears to be outdated (e.g. Winter Wren instead of Pacfic Wren).
library(dplyr)
View(full_join(species_eBird, species_JO))
