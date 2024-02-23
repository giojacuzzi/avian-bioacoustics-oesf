# Question 2A

plant_data = data.frame(
  species=c('A','B','C','D','E','F','G','H','I','J','K'),
  root_depth=c('shallow','shallow','shallow','shallow','shallow','medium','medium','medium','medium','deep','deep'),
  uptake=c('nitrate','nitrate','ammonium','glycine','N-fixer','ammonium','glycine','nitrate','nitrate','N-fixer','nitrate')
)

patch_data = list(
  c('A','B','C'),
  c('D','F','E','B'),
  c('C','A','F'),
  c('A','C','J','E'),
  c('H','J','F','A','B','D','E','C'),
  c('F','E','D'),
  c('C','B','E','D','A'),
  c('F','G','I','B','H'),
  c('G','B','A','J','K')
)

results = data.frame()

for (p in 1:length(patch_data)) {
  patch_species = patch_data[[p]]
  patch_plant_data = plant_data[plant_data$species %in% patch_species, ]
  
  # Unique functional traits
  unique_functional_traits = c(unique(patch_plant_data$root_depth), unique(patch_plant_data$uptake))
  message(length(unique_functional_traits), ' unique functional traits')
  print(unique_functional_traits)
  
  # For each N-fixer, find how many uptake pathways (other than the N-fixer) are in the same rooting depth
  n_fixer_uptakes = 0
  if (any(patch_plant_data$uptake == 'N-fixer')) {
    
    for (n in which(patch_plant_data$uptake == 'N-fixer')) {
      rd = patch_plant_data[n, 'root_depth']
      temp_patch_plant_data = patch_plant_data[-c(n), ]
      uptakes = unique(temp_patch_plant_data[temp_patch_plant_data$root_depth == rd, 'uptake'])
      message(length(uptakes), ' uptakes at depth ', rd)
      print(uptakes)
      n_fixer_uptakes = n_fixer_uptakes + length(unique(uptakes))
    }
  }
  
  biomass = length(unique_functional_traits) + n_fixer_uptakes * 0.5
  
  message('Biomass: ', biomass)
  
  results = rbind(results, data.frame(
    Patch       = p,
    Traits      = length(unique_functional_traits),
    NfixUptakes = n_fixer_uptakes,
    Biomass     = biomass
  ))
}

print(results)

# Question 2 B and C
library(ggplot2)

alpha_v_biomass = data.frame(
  Patch   = results$Patch,
  Alpha   = sapply(patch_data, length),
  Biomass = results$Biomass
)

ggplot(data = alpha_v_biomass, aes(x=Alpha, y=Biomass)) +
  geom_point() +
  stat_smooth(method = 'lm', formula = y ~ x, col='red', fill='red') +
  stat_smooth(method = 'lm', formula = y ~ log(x), col='blue', fill='blue') +
  labs(x = 'Alpha diversity', y = 'Biomass') +
  theme_minimal()

model_linear = lm(data = alpha_v_biomass, Biomass ~ Alpha)
model_log = lm(data = alpha_v_biomass, Biomass ~ log(Alpha))

summary(model_linear)$r.squared
summary(model_log)$r.squared


