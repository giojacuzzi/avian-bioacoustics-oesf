source('global.R')
library(ggplot2)
library(lme4)
library(faraway)
theme_set(theme_minimal())

#################################################
# Model diagnostics functions
# Are errors independent? -------------------------------------------------------------------
check_err_independent = function(model) {
  acf(residuals(model), type = 'correlation')
}
# it appears so

# Are errors identically distributed, i.e. constant/homogeneous variance? -----------------------
check_err_identically_dist = function(model) {
  plot(fitted(model), residuals(model),
       xlab = "Fitted Values", ylab = "Residuals",
       ylim = c(-max(residuals(model)),max(residuals(model))))
  abline(h=0)
  model_test = lm(I(sqrt(abs(residuals(model)))) ~ I(fitted(model)))
  sumary(model_test)
  print(plot + geom_abline(aes(intercept = coef(model_test)[1], slope = coef(model_test)[2]), color = 'red'))
  abline(model_test, col='red') # a significant slope indicates unequal variances
  # Serious correlation of errors generally requires a structural change in the model
  # Maybe as simple as adding a predictor (covariates), or may be more complex: generalized least squares or other approaches
  
  # Also perform Levene's Test for homogeneous variance - robust to non-normality
  group = rep(0, nobs(model))
  group[which(residuals(model)>median(residuals(model)))] <- 1
  group = as.factor(group)
  leveneTest(residuals(model), group) # p=value is small, errors are not homogenous!
}

# Are errors normally distributed? -------------------------------------------------------------
check_err_norm_dist = function(model) {
  hist(residuals(model))
  qqnorm(residuals(model))
  qqline(residuals(model))
}

#################################################

# Get data
joined_data = get_joined_data()

joined_data = na.omit(joined_data[joined_data$DataYear %in% c(2021), ])

# Get index data
data_acidx = read.csv('acoustic_indices/output/6am/results.csv')
data_acidx$SurveyDate = as.Date(data_acidx$SurveyDate)
data_acidx$SerialNo = factor(data_acidx$SerialNo)
data_acidx = left_join(data_acidx,
                       distinct(joined_data[, c('WatershedID', 'StationName_AGG',
                                         'SurveyDate', 'SerialNo', 'Strata', 'TempMax')])
                       , by = c('SurveyDate', 'SerialNo'))
data_acidx$Month = factor(month(data_acidx$SurveyDate))
data_acidx$Year = factor(year(data_acidx$SurveyDate))
data_raw = data_acidx[data_acidx$Year==2021,] # only look at 2021
data_raw = na.omit(data_raw)

data_noise = read.csv('qerm-514/output/annotations.csv')
data_noise$Noise = factor(data_noise$Noise)
data_noise$SurveyDate = as.Date(data_noise$SurveyDate)
data_raw = left_join(data_raw,
                       distinct(data_noise[, c('Noise','SurveyDate', 'SerialNo', 'Note')])
                       , by = c('SurveyDate', 'SerialNo'))

# Get weather/atmospheric data
data_atmo = read.csv('data/weather.csv')
data_atmo$datetime = as.POSIXct(sub('T', ' ', data_atmo$datetime), format = '%Y-%m-%d %H:%M:%S', tz=tz)
data_atmo$SurveyDate = as.Date(data_atmo$datetime)
data_atmo = data_atmo[hour(data_atmo$datetime)==6, ]
data_raw = left_join(data_raw,
                          distinct(data_atmo[, c('SurveyDate', 'temp','humidity', 'precip', 'windgust', 'windspeed', 'cloudcover', 'conditions')]), by = c('SurveyDate'))

# Create dataset without noisy dates
# data_acidx = data_acidx[!data_acidx$Noise=='rain' & !data_acidx$Noise=='wind' & !data_acidx$Noise=='other', ]
data = data_raw[data_raw$Noise=='', ]
data_noisy = data_raw

############################################################################################################
## Visualize data -------------------------------------------------------------------

# Date and strata vs acdix
ggplot(data_raw, aes(x = SurveyDate, y = ACI, color = Strata)) +
  geom_point() + labs(title = 'ACI per strata over time')
ggplot(data_raw, aes(x = SurveyDate, y = BIO, color = Strata)) +
  geom_point() + labs(title = 'BIO per strata over time')

# Strata vs acidx
ggplot(data_raw, aes(x = Strata, y = ACI, color = Strata)) +
  geom_boxplot() + labs(title = 'ACI x strata')
ggplot(data_raw, aes(x = Strata, y = BIO, color = Strata)) +
  geom_boxplot() + labs(title = 'BIO x strata')

# Plot temperature data
ggplot(joined_data, aes(x = SurveyDate, y = TempMax, color = Strata)) +
  geom_point() + labs(title = 'Temperature per strata')

# Temp vs acidx (naive linear regression)
ggplot(data_raw, aes(x = TempMax, y = ACI, color = Strata)) +
  geom_point() + labs(title = 'ACI x temp per strata') + geom_smooth(method='lm')
ggplot(data_raw, aes(x = TempMax, y = BIO, color = Strata)) +
  geom_point() + labs(title = 'BIO x temp per strata') + geom_smooth(method='lm')

# Initial full LMM with all observations
aci_lmm_noisy = lmer(
  ACI ~ Strata + Month + TempMax + TempMax*Strata + (1|WatershedID:StationName_AGG),
  data = data_noisy
)
sumary(aci_lmm_noisy)
check_err_identically_dist(aci_lmm_noisy)

bio_lmm_noisy = lmer(
  BIO ~ Strata + Month + TempMax + TempMax*Strata + (1|WatershedID:StationName_AGG),
  data = data_noisy
)
sumary(bio_lmm_noisy)
check_err_identically_dist(bio_lmm_noisy)

# Distribution of response variables
hist(data_noisy$ACI)
hist(data_noisy$BIO)

# Initial full GLMM with all observations for ACI
aci_glmm_noisy = glmer(
  ACI ~ Strata + Month + TempMax + TempMax*Strata + (1|WatershedID:StationName_AGG),
  data = data_noisy,
  family = Gamma(link='log'),
  nAGQ = 0 # only way to get convergence (less accurate) 1 is Laplace, >1 Gauss-Hermite
)
sumary(aci_glmm_noisy)

check_err_identically_dist(aci_glmm_noisy) # no
check_err_norm_dist(aci_glmm_noisy) # heavy skew in upper quantiles
check_err_independent(aci_glmm_noisy) # noisy recordings are not independent!

# Initial full GLMM with all observations for BIO
bio_glmm_noisy = glmer(
  BIO ~ Strata + Month + TempMax + TempMax*Strata + (1|WatershedID:StationName_AGG),
  data = data_noisy,
  family = Gamma(link='log'),
  nAGQ = 0 # only way to get convergence (less accurate) 1 is Laplace, >1 Gauss-Hermite
)
sumary(bio_glmm_noisy)

check_err_identically_dist(bio_glmm_noisy) # no, some structure present
check_err_norm_dist(bio_glmm_noisy) # slight skew
check_err_independent(bio_glmm_noisy) # noisy recordings are not independent!

################################################################################################
# Removing rain noise data ----------------------------------------------------------------------

hist(data_noisy$ACI)

# SHOW EXAMPLE .WAV FILES

# Plot date and strata vs acdix
ggplot(data_raw, aes(x = SurveyDate, y = ACI, color = Strata, shape = Noise)) +
  geom_point() + labs(title = 'ACI per strata over time') # rainy observations have high ACI because of extreme amplitude fluctuations
ggplot(data_raw, aes(x = SurveyDate, y = BIO, color = Strata, shape = Noise)) +
  geom_point() + labs(title = 'BIO per strata over time') # rainy observations have relatively low BIO because of homogeneous PSD

library(data.table)
l <- list(data_noisy, data)
names(l) <- c('With rain', 'Without rain')
zb <- rbindlist(l, id='id')

# Plot ACI by strata
ggplot(zb, aes(x=Strata, y=ACI, color=Strata)) +
  geom_boxplot() + facet_wrap(~id)
ggplot(zb, aes(x=Strata, y=BIO, color=Strata)) +
  geom_boxplot() + facet_wrap(~id)

# Plot naive linear regression by strata
ggplot(zb, aes(x=TempMax, y=ACI, color=Strata)) +
  geom_point() + geom_smooth(method='lm') + facet_wrap(~id) + labs(title = 'ACI x temp per strata')
ggplot(zb, aes(x=TempMax, y=BIO, color=Strata)) +
  geom_point() + geom_smooth(method='lm') + facet_wrap(~id) + labs(title = 'BIO x temp per strata')

# Distribution of response variables
hist(data_noisy$ACI)
hist(data$ACI)
hist(data_noisy$BIO)
hist(data$BIO)

####################################################################

# Look at data
# plot(data_raw$WatershedID)
# plot(data_raw$StationName_AGG)
# plot(data_raw$Month)

# Make your assumptions on the data clear, particularly for the acidx
# Hypothesis "I think the world works like this" > "If that's true, I expect to see this in the data"

# TODO: AIC model selection, cross validation?
# See slide 24 "Model selection" -- OR just skip steps 1 and 2 altogther, because it makes sense for your design
# Let's assume the random effect is necessary based on our sampling design, but make sure the formula is correct!
# Keep random effects as is and fit different fixed effects (using AIC)
# Be honest about the process you use here!

# TODO: points of influence? outliers?

# Precipitation / rain as categorical variable?
# Early and late breeding season?

# Potential variables: TempMax, TempMax*Strata, Month, etc.
# Fit a generalized linear mixed model
# Note that this follows similar methods to GLMs and LMMs

glmm_null = glmer(
  ACI ~ (1|WatershedID:StationName_AGG),
  data = data,
  family = Gamma(link='log'), # distribution of data y~f(y), and link function
  nAGQ = 1
) 
glmm_full = glmer(
  ACI ~ Strata + Month + TempMax + (1|WatershedID:StationName_AGG),
  data = data,
  family = Gamma(link='log'), # distribution of data y~f(y), and link function
  nAGQ = 0 # only way to get convergence (less accurate); 1 is Laplace, >1 Gauss-Hermite
)
glmm_notemp = glmer(
  ACI ~ Strata + Month + (1|WatershedID:StationName_AGG),
  data = data,
  family = Gamma(link='log'),
  nAGQ = 1 # Laplace
)
data$ACI_scaled = data$ACI - min(data$ACI) + 1 # TODO: normalize to noise floor ACI
glmm_scaled = glmer(
  ACI_scaled ~ Strata + Month + TempMax + (1|WatershedID:StationName_AGG),
  data = data, # + scale(TempMax)*Month ???
  family = Gamma(link='log'),
  nAGQ = 0
)
glmm_scaled_transf = glmer(
  log(ACI_scaled+0.1) ~ Strata + Month + TempMax + (1|WatershedID:StationName_AGG),
  data = data, # + scale(TempMax)*Month ???
  family = Gamma(link='log'),
  nAGQ = 0
)
lmm_scaled_transf = lmer(
  log(ACI_scaled) ~ Strata + Month + scale(TempMax) + (1|WatershedID:StationName_AGG),
  data = data
)

# AIC values
extractAIC(glmm_null)
extractAIC(aci_glmm_noisy)
extractAIC(glmm_full)
extractAIC(glmm_notemp)
extractAIC(glmm_scaled)
extractAIC(glmm_scaled_transf)
extractAIC(lmm_scaled_transf)

# Goodness of fit via chi-squared -- am I doing this correctly?
X2 <- sum((data$ACI - fitted(glmm_full))^2 / fitted(glmm_full))
pchisq(X2, df = nrow(data) - length(coef(glmm_full)), lower.tail = F) # if the p-value is large we assume the model is a good fit

# Are errors independent?
check_err_independent(aci_glmm_noisy)
check_err_independent(glmm_full) # it appears so
check_err_independent(glmm_scaled)
check_err_independent(glmm_scaled_transf)
check_err_independent(lmm_scaled_transf)

# Are errors identically distributed, i.e. constant/homogeneous variance?
check_err_identically_dist(aci_glmm_noisy, data_noisy)
check_err_identically_dist(glmm_full) # no, errors are heteroscedactic; p-value is small, errors not homogenous!
check_err_identically_dist(glmm_scaled)
check_err_identically_dist(glmm_scaled_transf)
check_err_identically_dist(lmm_scaled_transf)

# Are errors normally distributed?
check_err_norm_dist(aci_glmm_noisy)
check_err_norm_dist(glmm_full) # heavy-tailed, i.e. compared to the normal distribution there is more data located at the extremes of the distribution and less data in the center of the distribution
check_err_norm_dist(glmm_scaled) # slight left skew (lower tail extended, upper reduced, relative to normal)
check_err_norm_dist(glmm_scaled_transf)
check_err_norm_dist(lmm_scaled_transf)

###################################

bio_glmm_full = glmer(
  BIO ~ Strata + Month + TempMax + TempMax*Strata + (1|WatershedID:StationName_AGG),
  data = data,
  family = Gamma(link='log'), # distribution of data y~f(y), and link function
  nAGQ = 0 # only way to get convergence (less accurate) 1 is Laplace, >1 Gauss-Hermite
)
bio_glmm_notemp = glmer(
  BIO ~ Strata + Month + (1|WatershedID:StationName_AGG),
  data = data,
  family = Gamma(link='log'),
  nAGQ = 1 # Laplace
)

# AIC values
extractAIC(bio_glmm_noisy)
extractAIC(bio_glmm_full)
extractAIC(bio_glmm_notemp)

# Are errors independent?
check_err_independent(bio_glmm_noisy)
check_err_independent(bio_glmm_full) # no?

# Are errors identically distributed, i.e. constant/homogeneous variance?
check_err_identically_dist(bio_glmm_noisy, data_noisy)
check_err_identically_dist(bio_glmm_full, data) # no, p-value is small, errors not homogenous!
check_err_identically_dist(bio_glmm_full, data, strata=T)

# Are errors normally distributed?
check_err_norm_dist(bio_glmm_noisy)
check_err_norm_dist(bio_glmm_full)

###################################

# Weighted least squares ?
# Goodness of fit via pearson's chi square statistic?

# How to test for outliers and influential points in GLMM if hat-matrix assumptions don't hold?
# For GLMMs the leverages depend on the estimated variance-covariance matrices of the random effects

# TODO: look into overdispersion, commonly because trials are not independent and/or their is homegeneity within our predicted levels. Note that the estimates of Beta are not affected, but the variance of Beta is, by overdispersion.

# Zero-truncated data?

# See GLMM methods (laplace approximation and gauss-hermite quadrature), see slide 50

#####