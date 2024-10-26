# Load required packages 
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, 
               tidyr, 
               janitor,
               vegan)

# Read in data 
sp_comm <- readr::read_csv("https://raw.githubusercontent.com/guysutton/CBC_coding_club/master/data_raw/species_abundance_matrix_ex.csv") %>%
  # Clean column names 
  janitor::clean_names()

# Check data entry 
dplyr::glimpse(sp_comm)

sac_raw <- sp_comm %>%
  # Remove site decsription variables 
  dplyr::select(-c(provinces, climatic_zones, site, season, haplotype)) %>%
  # Compute SAC
  vegan::poolaccum(.)

# Extract observed richness (S) estimate 
obs <- data.frame(summary(sac_raw)$S, check.names = FALSE)
colnames(obs) <- c("N", "S", "lower2.5", "higher97.5", "std")
head(obs)

obs %>%
  ggplot(data = ., aes(x = N,
                       y = S)) +
  # Add confidence intervals
  geom_ribbon(aes(ymin = lower2.5,
                  ymax = higher97.5),
              alpha = 0.5,
              colour = "gray70") +
  # Add observed richness line 
  geom_line() +
  labs(x = "No. of surveys",
       y = "Observed richness",
       subtitle = "More surveys are required to find all the insects on this plant")
