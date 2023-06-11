# Gio Jacuzzi - QERM 514 final project
library(lme4)
library(ggplot2)
library(glmmTMB)
library(DHARMa)
library(lubridate)
library(faraway)
library(car)
library(dplyr)

# Hypothesis ############################################################################

# TODO: songbird activity
# Make your assumptions on the data clear, particularly for the acidx
# Hypothesis "I think the world works like this" > "If that's true, I expect to see this in the data"

# Hypothesis #1: Commercial thinning of forest stands provides more favorable habitat conditions for songbirds (while also promoting tree growth and quality for economic returns).
# Prediction #2: If this is true, we expect to see greater abundance in mid-successional stands that have been thinned, compared to those that have not."

# How can we quantify abundance at scale? Songbirds are very conspicuous creatures, particularly during the breeding season. They vocalize to establish territories, initiate courtship, and communicate. As such, vocalization activity reflects songbird community activity. The acoustic complexity index (ACI), a measure of bioacoustic activity within a frequency band, has been shown to be extremely correlated with number of avian vocalizations, and positively correlated with avian abundance in temperate forests. Here, ACI values are given on a continuous unbounded scale increasing in magnitude with increasing activity, where 0 corresponds to no activity, or the ambient "noise floor" of a soundscape.

# So, if commercial thinning provides better songbird habitat, we expect to see greater ACI.

# Hypothesis #2: Vocalization activity is influenced by atmospheric conditions. Specifically, activity is reduced during periods of high temperature, precipitation, cloudcover, and wind. -- OR temperature alone?
# Prediction #2: If this is true, each of these covariates should exhibit a negative effect on ACI.

# Other research questions:
# "Vocalization activity of songbird communities reflects known patterns in breeding activity across forest habitats. In other words, peaks in songbird vocalization reflect known peaks in the breeding season."
# << Note that this needs to be further validated by quantifying species-specific calls
# "Community vocalization activity corresponds to peaks in breeding activity -- early season (April) for several migratory species, and many year-round species (and some migratory species) in mid season (May-July).

## Load, preprocess, and visualize data #################################################

# Each row in the dataset corresponds to a dawn chorus observation at one site:
data_raw = read.csv('qerm-514/output/qerm514.csv')
data_raw$watershed = factor(data_raw$watershed)
data_raw$site = factor(data_raw$site)
data_raw$date = as.Date(data_raw$date)
data_raw$month = factor(month(data_raw$date))
data_raw$noise = factor(data_raw$noise)
data_raw$stage = factor(data_raw$stage, levels = c('Early', 'Mid', 'Mid (Thinned)', 'Late'))

# Visualize the data
theme_set(theme_minimal())
stage_colors = c('#A25B5B', '#A4BE7B', '#54436B', '#285430')
stage_colors = c('#73E2A7', '#1C7C54', '#1C7C54', '#6c584c')

p = ggplot(data_raw, aes(x = date, y = aci, color = stage)) + geom_point() +
  scale_color_manual(values = stage_colors) + labs(title = 'ACI by strata over the breeding season', x = 'Date', y = 'ACI', color = 'Forest stage')
p
ggsave(p + theme(text = element_text(size = 28), plot.margin = margin(1,1,1,1, 'cm')),
       file='qerm-514/output/aci_season.png', width=16, height=12)

p = ggplot(data_raw, aes(x = month, y = aci)) + geom_boxplot() + labs(title = 'ACI by month', x = 'Month', y = 'ACI')
p
ggsave(p + theme(text = element_text(size = 28), plot.margin = margin(1,1,1,1, 'cm')),
       file='qerm-514/output/aci_month.png', width=12, height=12)

hist(data_raw$aci, main = 'Histogram of ACI', xlab = 'ACI', ylab = 'Frequency')
p = ggplot(data_raw, aes(x=aci)) + geom_histogram(binwidth = 20) + labs(title = 'Histogram of ACI', x = 'ACI', y = 'Count')
p
ggsave(p + theme(text = element_text(size = 28), plot.margin = margin(1,1,1,1, 'cm')),
       file='qerm-514/output/aci_hist.png', width=12, height=12)

# Things that we notice:
# - The grouping factors of sites within watersheds, and inherent correlation of multiple observations per site, suggests a random effect structure is needed to avoid pseudoreplication, suggesting a mixed model is needed.
# - Distribution of response variable is heavily skewed, with most values being between (0-50]. This suggests either a transformation or a generalized model is needed, following a gamma distribution and log link function.
# - We have some potential outliers

# Exploring potential outliers. Note that with mixed models we cannot use the hat matrix to find outliers or points of influence, so we are left with looking at a subset of dimensions for outliers (e.g. normal boxplots). It turns out the outliers are mostly days with heavy rain or wind gusts, which the ACI algorithm quantifies as high acoustic activity in the frequency band of songirds:

data_raw$audible_rain = (data_raw$noise == 'rain')
ggplot(data_raw, aes(x = date, y = aci, color = audible_rain)) + geom_point() +
  scale_color_manual(values = c('black', 'blue')) + labs(title = 'ACI by strata over the breeding season')

# Filter out observations with audible rain in songbird frequency band
data = data_raw[data_raw$audible_rain==F, ]

hist(data$aci)

hist(data$aci, main = 'Histogram of ACI (audible outliers removed)', xlab = 'ACI', ylab = 'Frequency')
p = ggplot(data, aes(x=aci)) + geom_histogram(binwidth = 20) + labs(title = 'Histogram of ACI', x = 'ACI', y = 'Count')
p
ggsave(p + theme(text = element_text(size = 28), plot.margin = margin(1,1,1,1, 'cm')),
       file='qerm-514/output/aci_hist_after.png', width=12, height=12)

# Compare ACI by month with and without outliers
library(data.table)
l = list(data_raw, data)
names(l) = c('With audible rain', 'Without audible rain')
p = ggplot(rbindlist(l, id='id'), aes(x=month, y=aci)) +
  geom_boxplot() + facet_wrap(~id) + labs(title = 'ACI by month')
p
ggsave(p + theme(text = element_text(size = 28), plot.margin = margin(1,1,1,1, 'cm')),
       file='qerm-514/output/aci_month_compare.png', width=12, height=12)

# Model design ##########################################################################

# What are our model parameters?
y = 'aci'
x_continuous  = c('cloudcover', 'humidity', 'precip', 'temp', 'tempmax', 'windgust', 'windspeed')
x_categorical = c('stage', 'month', 'watershed', 'site')

# Note that we treat month as a categorical variable because we expect it will not have a linear effect across the breeding season

# Look at correlations (encode categorical variables as integers)
library(ggcorrplot)
data_int_encoded = data[, c(y, x_continuous, x_categorical)]
data_int_encoded[, x_categorical] = lapply(data_int_encoded[, x_categorical], as.numeric)
model.matrix(~0+., data=data_int_encoded) %>% 
  cor(use='pairwise.complete.obs') %>% 
  ggcorrplot(show.diag=FALSE, type='lower', lab=T, lab_size=4)

# Note that site and watershed are extremely correlated, they are nearly describing the exact same thing. TODO: We opt to only use watershed instead of a nested effect of site within watershed, which would significantly increase model complexity. In the future, we could compare results from PBmodcomp instead of choosing arbitrarily.

# Let's scale our continuous fixed effects. Note that they now have mean at 0 and variance = standard deviation = 1. If we want a prediction of acoustic activity at the mean temperature, we predict at the mean value, i.e. temp = 0. Note that scaling changes our interpretation of betas.
data[, x_continuous] = scale(data[, x_continuous])

# Let's start by fitting a general global model with all possible fixed effects included using Gauss-Hermite Quadrature
cntrl = glmerControl(optimizer = 'bobyqa', tol = 1e-4, optCtrl=list(maxfun=1000000))
global_glmm = glmer(
  aci ~ stage + month + cloudcover + humidity + precip + temp + tempmax + windgust + windspeed + (1|watershed:site),
  data = data, control = cntrl, family = Gamma(link='log'), nAGQ = 0
)

# Using glmmTMB?
global_glmmTMB = glmmTMB( # Laplace
  aci ~ stage + month + cloudcover + humidity + precip + temp + tempmax + windgust + windspeed + (1|watershed:site),
  data = data, family = Gamma(link='log')
)

global_lmm = lmer(
  log(aci) ~ stage + month + cloudcover + humidity + precip + temp + tempmax + windgust + windspeed + (1|watershed:site),
  data = data
)

# Diagnostics of model assumptions

check_err_identically_dist = function(model) {
  plot(fitted(model), residuals(model),
       xlab = "Fitted Values", ylab = "Residuals",
       ylim = c(-max(residuals(model)),max(residuals(model))))
  abline(h=0)
  model_test = lm(I(sqrt(abs(residuals(model)))) ~ I(fitted(model)))
  sumary(model_test)
  abline(model_test, col='red') # a significant slope indicates unequal variances

  # Also perform Levene's Test for homogeneous variance - robust to non-normality
  group = rep(0, nobs(model))
  group[which(residuals(model)>median(residuals(model)))] <- 1
  group = as.factor(group)
  leveneTest(residuals(model), group)
}

check_err_norm_dist = function(model) {
  qqnorm(residuals(model))
  qqline(residuals(model))
}

# Use "simulateResiduals" from DHARMa to do some diagnostics (using standardized residuals)
# See https://cran.r-project.org/web/packages/DHARMa/vignettes/DHARMa.html
n = 1000
sim_glmer = simulateResiduals(global_glmm, n)
sim_glmmTMB = simulateResiduals(global_glmmTMB, n)
sim_lmer = simulateResiduals(global_lmm, n)

par(mfrow = c(3, 1))
testOutliers(sim_glmer)
testOutliers(sim_glmmTMB)
testOutliers(sim_lmer)

# QQ plot of residuals (expected vs observed)
par(mfrow = c(1, 3))
testUniformity(sim_glmer)
testUniformity(sim_glmmTMB)
testUniformity(sim_lmer)

# Residuals vs predictions
par(mfrow = c(1, 3))
testQuantiles(sim_glmer)
testQuantiles(sim_glmmTMB)
testQuantiles(sim_lmer)

# Observed vs fitted values
par(mfrow = c(1, 3))
plot(data$aci, fitted(global_glmm), main='glmer', xlab = 'Observed', ylab = 'Fitted')
abline(a=0, b=1, col='red')
plot(data$aci, fitted(global_glmmTMB), main='glmmTMB', xlab = 'Observed', ylab = 'Fitted')
abline(a=0, b=1, col='red')
plot(data$aci, exp(fitted(global_lmm)), main='lmer (log transformed)', xlab = 'Observed', ylab = 'Fitted')
abline(a=0, b=1, col='red')

# Are errors identically distributed, i.e. constant/homogeneous variance?
check_err_identically_dist(global_glmm); title('GLMM')
check_err_identically_dist(global_glmmTMB); title('GLMM TMB')
check_err_identically_dist(global_lmm); title('LMM (log transform)')

# Are errors normally distributed?
check_err_norm_dist(global_glmm)
check_err_norm_dist(global_glmmTMB)
check_err_norm_dist(global_lmm)

# Are errors independent? Test temporal autocorrelation (Durbin-Watson), aggregating residuals by time
testTemporalAutocorrelation(
  recalculateResiduals(sim_glmer, group = data$date),
  time = unique(data$date)
)
testTemporalAutocorrelation(
  recalculateResiduals(sim_glmmTMB, group = data$date),
  time = unique(data$date)
)
testTemporalAutocorrelation(
  recalculateResiduals(sim_lmer, group = data$date),
  time = unique(data$date)
)

# Can also look at temporal autocorrelation of a specific site
site_to_test = (data$site == data$site[1])
testTemporalAutocorrelation(
  recalculateResiduals(sim_glmer, sel = site_to_test),
  time = unique(data[site_to_test, 'date'])
)

# "The gamma distribution assumes a specific pattern of heteroscedasticity in which the variance increases proportionally with the mean — specifically, the square of the mean. These assumptions can be assessed using deviance residuals (which are analogous to residual sum of squares in ordinary least squares [OLS]). Residuals should tend towards normality and homoscedasticity for continuous responses."

# Distribution of random effects looks normal
qqnorm(unlist(ranef(global_glmm)$`watershed:site`), main = 'QQ plot (REs)', pch = 16)
qqline(unlist(ranef(global_glmm)$`watershed:site`))

qqnorm(unlist(ranef(global_glmmTMB)), main = 'QQ plot (REs)', pch = 16)
qqline(unlist(ranef(global_glmmTMB)))

qqnorm(unlist(ranef(global_lmm)$`watershed:site`), main = 'QQ plot (REs)', pch = 16)
qqline(unlist(ranef(global_lmm)$`watershed:site`))

# Assuming our GLMM model passes diagnostic assumptions (global_glmmTMB), let's test our fixed effects
# models = list(
#   model_NULL = glmmTMB(
#     aci ~ (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s = glmmTMB(
#     aci ~ stage + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_m = glmmTMB(
#     aci ~ month + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_c_h_p_t_tm_wg_ws = glmmTMB(
#     aci ~ stage + month + cloudcover + humidity + precip + temp + tempmax + windgust + windspeed + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_c_h_p_t_tm_wg_ws = glmmTMB(
#     aci ~ stage + cloudcover + humidity + precip + temp + tempmax + windgust + windspeed + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_c_h_p_t_tm_ws = glmmTMB(
#     aci ~ stage + month + cloudcover + humidity + precip + temp + tempmax + windspeed + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_c_h_p_t_tm_wg = glmmTMB(
#     aci ~ stage + month + cloudcover + humidity + precip + temp + tempmax + windgust + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_c_h_p_t_wg_ws = glmmTMB(
#     aci ~ stage + month + cloudcover + humidity + precip + temp + windgust + windspeed + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_c_h_p_tm_wg_ws = glmmTMB(
#     aci ~ stage + month + cloudcover + humidity + precip + tempmax + windgust + windspeed + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_c_h_p_tm_ws = glmmTMB(
#     aci ~ stage + month + cloudcover + humidity + precip + tempmax + windspeed + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m = glmmTMB(
#     aci ~ stage + month + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_c = glmmTMB(
#     aci ~ stage + month + cloudcover + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_h = glmmTMB(
#     aci ~ stage + month + humidity + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_p = glmmTMB(
#     aci ~ stage + month + precip + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_t = glmmTMB(
#     aci ~ stage + month + temp + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_tm = glmmTMB(
#     aci ~ stage + month + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_wg = glmmTMB(
#     aci ~ stage + month + windgust + (1|site),
#     data = data, family = Gamma(link='log')
#   ),
#   model_s_m_ws = glmmTMB(
#     aci ~ stage + month + windspeed + (1|site),
#     data = data, family = Gamma(link='log')
#   )
# )

models = list(
  model_NULL = glmer(
    aci ~ (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s = glmer(
    aci ~ stage + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_m = glmer(
    aci ~ month + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_c_h_p_t_tm_wg_ws = glmer(
    aci ~ stage + month + cloudcover + humidity + precip + temp + tempmax + windgust + windspeed + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_c_h_p_t_tm_wg_ws = glmer(
    aci ~ stage + cloudcover + humidity + precip + temp + tempmax + windgust + windspeed + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_c_h_p_t_tm_ws = glmer(
    aci ~ stage + month + cloudcover + humidity + precip + temp + tempmax + windspeed + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_c_h_p_t_tm_wg = glmer(
    aci ~ stage + month + cloudcover + humidity + precip + temp + tempmax + windgust + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_c_h_p_t_wg_ws = glmer(
    aci ~ stage + month + cloudcover + humidity + precip + temp + windgust + windspeed + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_c_h_p_tm_wg_ws = glmer(
    aci ~ stage + month + cloudcover + humidity + precip + tempmax + windgust + windspeed + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_c_h_p_tm_ws = glmer(
    aci ~ stage + month + cloudcover + humidity + precip + tempmax + windspeed + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m = glmer(
    aci ~ stage + month + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_c = glmer(
    aci ~ stage + month + cloudcover + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_h = glmer(
    aci ~ stage + month + humidity + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_p = glmer(
    aci ~ stage + month + precip + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_t = glmer(
    aci ~ stage + month + temp + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_tm = glmer(
    aci ~ stage + month + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_wg = glmer(
    aci ~ stage + month + windgust + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  ),
  model_s_m_ws = glmer(
    aci ~ stage + month + windspeed + (1|site),
    data = data, family = Gamma(link='log'), control = cntrl, nAGQ = 20
  )
)

model_selection = t(data.frame(sapply(models, extractAIC)))
colnames(model_selection) = c('edf', 'AIC')
model_selection = data.frame(model_selection)
model_selection = model_selection[order(model_selection$AIC),]
model_selection

# Examine the general effects of atmospheric covariates
large_model = models['model_s_m_c_h_p_tm_ws'][[1]]
fixef(large_model)
# confint(large_model, level = 0.95, method = 'profile')
# bci <- confint(large_model,method="boot",nsim=200)
# bci

# It appears a model with only stage and month is optimal
model = models[ rownames(model_selection)[1] ][[1]]
summary(model)
VarCorr(model) # NOTE: doesn't work with glmmTMB
fixef(model)
# confint(model, level = 0.95, method = 'profile')
# bci <- confint(model,method="boot",nsim=200)
# bci

# The multiplicative increase in ACI for thinned compared to non-thinned (the exponential function of Beta1)
# In other words, thinned stands have ~1.5 times more ACI compared to non-thinned.
exp(fixef(model)[2])

# Findings:
# - As expected, forest stage habitat has the greatest effect on activity.
# - Thinned stands do improve habitat! So much so that they more closely resemble the acoustic activity of mature forests than they do non-thinned stands of the same age. (Show in smoothed mean ACI by month graph too, with dotted line for thinned)
# - Interestingly, the mean activity of early-stage forests across the breeding season is nearly TODO times more than the other habitats
# - Peaks in activity occur during April and June, which correspond with known breeding schedules
# - It appears that, generally, ACI is negatively affected by increasing atmospheric conditions, but the confidence intervals of parameter estimates do not exclude zero, so this is speculative.

# Beta0 is the ACI prediction for mid-seral habitat in April (the start of the breeding season) with mean observed cloudcover
# Beta1 is the difference between thinned mid-seral and non-thinned mid-seral habitat (the same across month and cloudcover)
# Beta2 and Beta 3 are the differences between between early and late-seral habitat, respectively (the same across season and cloudcover)
# Beta4, 5, and 6 are the differences between May, June, and July, respectively, and April (the same across habitat)
# Beta7 is the predicted change in ACI for a one-unit change in scaled cloudcover, which is equal to a change of 1 SD in observed cloudcover

data$thinned = (data$stage == 'Mid (Thinned)')

p = ggplot(data, aes(x = stage, y = aci, fill = stage)) + geom_boxplot() + scale_fill_manual(values = stage_colors) + labs(title = 'ACI by forest stage', x = 'Forest Stage', y = 'ACI', fill = 'Forest Stage')
p
ggsave(p + theme(text = element_text(size = 28), plot.margin = margin(1,1,1,1, 'cm')),
       file='qerm-514/output/aci_stage.png', width=12, height=12)

p = ggplot(data %>% group_by(stage, month) %>% summarise_at(vars('aci'), mean), aes(x = month, y = aci, color = stage, group = stage)) + geom_smooth(lwd=2) + scale_color_manual(values = stage_colors) + labs(title = 'Mean ACI by stage by month', x = 'Month', y = 'ACI', color = 'Forest Stage')
p
ggsave(p + theme(text = element_text(size = 28), plot.margin = margin(1,1,1,1, 'cm')),
       file='qerm-514/output/mean_aci_stage_month.png', width=15, height=12)

data$week = factor(strftime(data$date, format = '%V'))
p = ggplot(data %>% group_by(stage, week) %>% summarise_at(vars('aci'), mean), aes(x = week, y = aci, color = stage, group = stage)) + geom_smooth(lwd=2) + scale_color_manual(values = stage_colors) + labs(title = 'Mean ACI by stage by week', x = 'Week', y = 'ACI', color = 'Forest Stage')
p

p = ggplot(data, aes(x = date, y = aci, color = stage, group = stage, fill = stage)) + geom_smooth(lwd=2) + scale_color_manual(values = stage_colors) + scale_fill_manual(values = stage_colors) + labs(title = 'ACI by stage over time', x = 'Date', y = 'ACI', color = 'Forest Stage')
p

ggplot(data %>% group_by(stage, date) %>% summarise_at(vars('aci'), mean), aes(x = date, y = aci, color = stage, group = stage, fill = stage)) + geom_line(lwd=1) + scale_color_manual(values = stage_colors) + scale_fill_manual(values = stage_colors) + labs(title = 'ACI by stage over time', x = 'Date', y = 'ACI', color = 'Forest Stage')

ggplot(data, aes(x = date, y = aci, color = stage, group = stage, fill = stage, linetype = thinned)) + geom_line(stat='smooth', method='gam', lwd=2, level=0.95) + scale_color_manual(values = stage_colors) + scale_fill_manual(values = stage_colors) + labs(title = 'Mean ACI by stage over time', x = 'Date', y = 'ACI', color = 'Forest Stage')

# Next steps:
# - Narrow down diagnostic goals for gamma GLMM
# - We often have multiple plausible models. Multiple models may be truly meaningful in terms of representing fundamental ecological hypotheses, or they may just deal with uncertainty about some nuisance variable (e.g., detection probability in animal studies) · Using a single model sweeps this uncertainty under the rug · Recognition of multiple models is the first step to incorporating model uncertainty in estimates · Single model measures of uncertainty: Accounts only for our uncertainty given that model · Model-averaged measures of uncertainty: Account for both uncertainty conditional on a model and uncertainty about which model (in the set) is best
# - Include additional covariates for elevation and aspect
# - Look at different times of day, besides dawn chorus
# - Look at species-specific results. Does the unique month bump in thinned correspond to a unique community composition compared to other habitats? Do these trends hold across species? How does species richness change?