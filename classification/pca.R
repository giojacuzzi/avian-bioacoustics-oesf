# setwd("~/Dropbox/Research/ESA_Julian_Analysis")

library(vegan)
library(FactoMineR)
library (factoextra)
library(fossil)

source("~/Dropbox/Research/ESA_Julian_Analysis/biostats.R", encoding = 'UTF-8')

####### Site analysis

# # "~/Dropbox/Research/ESA_Julian_Analysis/birdcount_average_per_site_oesf.csv"
sitedata <- read.csv('classification/_output/birdcount_average_per_site_oesf.csv', header = T, row.names=1) # "~/Dropbox/Research/ESA_Julian_Analysis/birdcount_average_per_site_oesf.csv"
sitedata = t(sitedata)
sites_to_strata <- read.csv("~/Dropbox/Research/ESA_Julian_Analysis/sites_to_strata.csv", header=T) # read.csv("classification/_output/sites_to_strata.csv", header = T) #
sites_to_strata$strata = recode_factor(as.factor(sites_to_strata$strata), `Stand Initiation` = 'Early', `Competitive Exclusion` = 'Mid', `Thinned` = 'Mid (Thinned)', `Mature` = 'Late')

sitedata <- drop.var(sitedata,min.po=1)

### DEBUGGING
# Set to occupancy 1 or 0
# sitedata[-1] <- as.integer(sitedata[-1] != 0)
#

sitedata<-decostand(sitedata, method='hel') # TODO

#### DEBUGGING
# Only look at early seral
# sitedata = sitedata[rownames(sitedata) %in% sites_to_strata[sites_to_strata$strata=='Early', 'site'], ]
####

bird.pca<-PCA(sitedata,graph=F)

theme_set(theme_minimal())

fviz_screeplot(bird.pca, addlabels = TRUE, ylim = c(0, 30))
fviz_pca_var(bird.pca, col.var="contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = F, title="PCA variables (species)", ggtheme = theme(text = element_text(size = 18)))

fviz_pca_biplot(bird.pca, label="var",col.var="contrib",repel = F, title="", ggtheme = theme(text = element_text(size = 18)))

strata_colors = c('#73E2A7', '#1C7C54',  '#73125A', '#6c584c')
fviz_pca_ind(bird.pca, geom = "point", habillage = sites_to_strata$strata, palette = strata_colors, 
             addEllipses = TRUE, ellipse.level=0.95,
             title="", ggtheme = theme(text = element_text(size = 18)))

# fviz_pca_ind(bird.pca, geom = "point", habillage = sites_to_strata$comp_early_seral, 
#              addEllipses = TRUE, ellipse.level=0.95,
#              title="", ggtheme = theme(text = element_text(size = 18)))

# PERMANOVA
spe.perm<-adonis2(sitedata~strata,data=sites_to_strata,permutations=1000,method='euclidean')
spe.perm

####### Site-date analysis
data <- read.csv("~/Dropbox/Research/ESA_Julian_Analysis/birdcounts_oesf.csv", header = T)
names(data)

df <- create.matrix(data, tax.name = "common_name",
                    locality = "sitedate",
                    abund.col = "count",
                    abund = TRUE)

birddata <- as.data.frame(t(df))

birddata <- drop.var(birddata,min.po=1)

birddata<-decostand(birddata, method='hel')

bird.pca<-PCA(birddata,graph=F)

write.csv(bird.pca$ind$coord,"site_date_PCAscores.csv")

# Plot like the others, but now each point is a site-date combination
fviz_screeplot(bird.pca, addlabels = TRUE, ylim = c(0, 30))
fviz_pca_var(bird.pca, col.var="contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = F, title="PCA variables (species)", ggtheme = theme(text = element_text(size = 18)))
fviz_pca_biplot(bird.pca, label="var",col.var="contrib",repel = F, title="", ggtheme = theme(text = element_text(size = 18)))

site_date_PCAscores <- as.data.frame((bird.sitedate.pca$ind)$coord[,1:2])
dates = sapply(strsplit(rownames(site_date_PCAscores), '_'), '[[', 2)
site_date_PCAscores$date = as.character(dates)
site_date_PCAscores$date = as.Date(site_date_PCAscores$date)
site_date_PCAscores$site = sapply(strsplit(rownames(site_date_PCAscores), '_'), '[[', 1)
site_date_PCAscores = full_join(site_date_PCAscores, sites_to_strata[,c('site', 'strata')], by = c('site'))

library(patchwork)
p1 = ggplot(site_date_PCAscores, aes(x = date, y = Dim.1, color = strata)) +
  geom_smooth(method = 'loess', span = 0.25, se = F) +
  # geom_line() +
  geom_point(shape=16, alpha=0.35) +
  scale_color_manual(name = 'Strata', values = strata_colors) +
  labs(title='Principal components 1 and 2', y = 'Dim 1') +
  annotate("rect",xmin=as.Date('2021-06-26'),xmax=as.Date('2021-06-29'),ymin=-Inf,ymax=Inf, alpha=0.1, fill='red') +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(), text = element_text(size = 18))
p1


