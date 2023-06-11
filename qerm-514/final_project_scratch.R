# Gio Jacuzzi - QERM 514 final project

library(DHARMa)
library(lme4)
library(faraway)
library(car)
library(ggplot2)
source('global.R')

# Model diagnostics functions #################################################################

# Are errors independent?
check_err_independent = function(model) {
  acf(residuals(model), type = 'correlation')
}

# Are errors identically distributed, i.e. constant/homogeneous variance?
check_err_identically_dist = function(model) {
  plot(fitted(model), residuals(model),
       xlab = "Fitted Values", ylab = "Residuals",
       ylim = c(-max(residuals(model)),max(residuals(model))))
  abline(h=0)
  model_test = lm(I(sqrt(abs(residuals(model)))) ~ I(fitted(model)))
  sumary(model_test)
  abline(model_test, col='red') # a significant slope indicates unequal variances
  # Serious correlation of errors generally requires a structural change in the model
  # May be as simple as adding a predictor (covariates),
  # or may be more complex: generalized least squares or other approaches
  
  # Also perform Levene's Test for homogeneous variance - robust to non-normality
  group = rep(0, nobs(model))
  group[which(residuals(model)>median(residuals(model)))] <- 1
  group = as.factor(group)
  leveneTest(residuals(model), group)
}

# Are errors normally distributed?
check_err_norm_dist = function(model) {
  # hist(residuals(model))
  qqnorm(residuals(model))
  qqline(residuals(model))
}

# Load data ####################################################################################

sub_years = c(2021) # only look at 2021
sub_months = c(4,5,6,7) # only look at April through July
joined_data = get_joined_data()
joined_data = na.omit(joined_data[joined_data$DataYear %in% sub_years, ])
joined_data$Month = factor(month(joined_data$SurveyDate))
joined_data = joined_data[joined_data$Month %in% sub_months, ]

# Get acoustic index data
data_raw = read.csv('acoustic_indices/output/6am/results.csv')
data_raw$SurveyDate = as.Date(data_raw$SurveyDate)
data_raw$SerialNo = factor(data_raw$SerialNo)
data_raw = left_join(
  data_raw,
  distinct(joined_data[, c('WatershedID', 'StationName_AGG', 'SurveyDate', 'SerialNo', 'Strata', 'TempMax')]), by = c('SurveyDate', 'SerialNo'))
data_raw$Month = factor(month(data_raw$SurveyDate))
data_raw$Year = factor(year(data_raw$SurveyDate))
data_raw = data_raw[data_raw$Year %in% sub_years,] # filter for sub_years
data_raw = data_raw[data_raw$Month %in% sub_months,] # filter for sub_months
data_raw = na.omit(data_raw)

# Get noise annotations data
data_noise = read.csv('qerm-514/output/annotations.csv')
data_noise$Noise = factor(data_noise$Noise)
data_noise$SurveyDate = as.Date(data_noise$SurveyDate)
data_raw = left_join(
  data_raw,
  distinct(data_noise[, c('Noise','SurveyDate', 'SerialNo', 'Note')]), by = c('SurveyDate', 'SerialNo'))

# Get weather/atmospheric data
data_atmo = read.csv('data/weather.csv')
data_atmo$datetime = as.POSIXct(sub('T', ' ', data_atmo$datetime), format = '%Y-%m-%d %H:%M:%S', tz=tz)
data_atmo$SurveyDate = as.Date(data_atmo$datetime)
data_atmo = data_atmo[hour(data_atmo$datetime)==6, ]
data_atmo['windgust'][is.na(data_atmo['windgust'])] = 0 
data_raw = left_join(
  data_raw,
  distinct(data_atmo[, c('SurveyDate', 'temp','humidity', 'precip', 'windgust', 'windspeed', 'cloudcover', 'conditions')]), by = c('SurveyDate'))

# Factor breeding season
data_raw$season_halves = cut(data_raw$SurveyDate, 2, labels=c('Early', 'Late'))
data_raw$season_thirds = cut(data_raw$SurveyDate, 3, labels=c('Early', 'Mid', 'Late'))
data_raw$week = factor(strftime(data_raw$SurveyDate, format = '%V'))

# Scale ACI
data_raw$aci = data_raw$ACI - round(min(data_raw$ACI)) # TODO: scale to noise floor as minimum, not the recorded minimum dawn chorus

# Rename columns
colnames(data_raw)[colnames(data_raw) == 'WatershedID'] = 'watershed'
colnames(data_raw)[colnames(data_raw) == 'StationName_AGG'] = 'site'
colnames(data_raw)[colnames(data_raw) == 'SurveyDate'] = 'date'
colnames(data_raw)[colnames(data_raw) == 'Strata'] = 'stage'
colnames(data_raw)[colnames(data_raw) == 'Month'] = 'month'
colnames(data_raw)[colnames(data_raw) == 'TempMax'] = 'tempmax'

# Reorder stage factor
levels(data_raw$stage)[levels(data_raw$stage) == 'STAND INIT'] = 'Early'
levels(data_raw$stage)[levels(data_raw$stage) == 'COMP EXCL'] = 'Mid'
levels(data_raw$stage)[levels(data_raw$stage) == 'THINNED'] = 'Mid (Thinned)'
levels(data_raw$stage)[levels(data_raw$stage) == 'MATURE'] = 'Late'
data_raw$stage = factor(data_raw$stage, levels = c('Mid', 'Mid (Thinned)', 'Early', 'Late'))
stage_colors = c('#A25B5B', '#A4BE7B', '#54436B', '#285430')

if (anyNA(data_raw)) stop('NAs in data!')

# Create dataset without noisy dates
data = data_raw[data_raw$Noise=='', ] # [rain, wind, other]
data_noisy = data_raw

# Investigate data ###############################################################################
theme_set(theme_minimal())

ggplot(data, aes(x = date, y = aci, color = stage)) + geom_point() +
  scale_color_manual(values = stage_colors) + labs(title = 'ACI by strata over the breeding season')

ggplot(data, aes(x = season_thirds, y = aci)) + geom_boxplot() + labs(title = 'ACI by season')

ggplot(data, aes(x = month, y = aci)) + geom_boxplot() + labs(title = 'ACI by season')

ggplot(data, aes(x = stage, y = aci, fill = stage)) + geom_boxplot() + scale_fill_manual(values = stage_colors) + labs(title = 'ACI by stage')

ggplot(data, aes(x = month, y = aci, fill = stage)) + geom_boxplot() + scale_fill_manual(values = stage_colors) + labs(title = 'ACI by stage by month')

ggplot(data %>% group_by(stage, month) %>% summarise_at(vars('aci'), mean), aes(x = month, y = aci, color = stage, group = stage)) + geom_smooth() + scale_color_manual(values = stage_colors) + labs(title = 'Mean ACI by stage by month')

ggplot(data %>% group_by(stage, month) %>% summarise_at(vars('aci'), mean), aes(x = month, y = aci, color = stage, group = stage)) + geom_line() + scale_color_manual(values = stage_colors) + labs(title = 'Mean ACI by stage by month')

ggplot(data, aes(x = date, y = tempmax, color = stage)) + geom_point() + scale_color_manual(values = stage_colors) + labs(title = 'Temperature by stage')

ggplot(data, aes(x = tempmax, y = aci, color = stage)) + geom_point() + scale_color_manual(values = stage_colors) + geom_smooth(method='lm') + labs(title = 'Temperature vs ACI by stage')


# Select variables from the dataset

# Note that we treat month as a categorical variable because we expect it will not have a linear effect across the breeding season
y = 'aci'
x_continuous  = c('tempmax', 'temp', 'windspeed', 'cloudcover')
x_categorical = c('stage', 'month', 'watershed', 'site')
data = data[, c(y, x_continuous, x_categorical)]

# Look at correlations
# Encode categorical variables as integers
data_int_encoded = data
data_int_encoded[, x_categorical] = lapply(data_int_encoded[, x_categorical], as.numeric)
correlation_matrix = cor(subset(data_int_encoded, select = c(y, x_continuous, x_categorical)))
correlation_matrix

library(ggcorrplot)
model.matrix(~0+., data=data_int_encoded) %>% 
  cor(use='pairwise.complete.obs') %>% 
  ggcorrplot(show.diag=FALSE, type='lower', lab=T, lab_size=4)
# note that site and watershed are extremely correlated, they are nearly describing the exact same thing, so just use watershed instead!

# Hypothesis ###################################################################################

# TODO: songbird activity
# Make your assumptions on the data clear, particularly for the acidx
# Hypothesis "I think the world works like this" > "If that's true, I expect to see this in the data"

# "Commercial thinning of forest stands provides more favorable habitat conditions for songbirds (while also promoting tree growth and quality for economic returns). If this is true, I expect to see greater vocalization activity in mid-successional stands that have been thinned, compared to those that have not."

# ACI has been shown to be positively correlated with avian abundance in temperate forests.

# "Vocalization activity of songbird communities reflects known patterns in breeding activity across forest habitats. In other words, peaks in songbird vocalization reflect known peaks in the breeding season."
# << Note that this needs to be further validated by quantifying species-specific calls
# "Community vocalization activity corresponds to peaks in breeding activity -- early season (April) for several migratory species, and many year-round species (and some migratory species) in mid season (May-July).

# "Atmospheric conditions of temperature, precipitation, cloudcover, and windspeed each have a negative effect on vocalization activity."

# A good analysis begins with a good question A good question is... An interesting question: it’s a question that relates to something that is relevant and important to basic science or management science. A question with an informative answer. By answering the question, we will be able to distinguish between hypotheses or in some other way provide critical information to address the interesting scientific issue.

# Model design #################################################################################

# Start with random effects. Think hard about your problem: are there some likely grouping factors in your data? Identify important grouping factors based on bootstrap LRT (or just leave them all in...)
# TODO: test random effects with pbkrtest

# We can follow these steps:
# 1. Fit a model with all of the possible fixed effects included
# 2. Keep the fixed effects constant and search for random effects (use bootstrap LRT)

# Once you’ve decided on your random effects, proceed to model selection for fixed effects (but don’t use REML when you do). How to structure the design matrix?

   # e.g. (1|pond:month) -> we estimate a random effect for each month at each pond - the effect of month can vary by pond

# Set control parameters 
cntrl = glmerControl(optimizer = 'bobyqa', tol = 1e-4, optCtrl=list(maxfun=1000000))

# Develop a full "global" model based on the distribution of your data (GLMM vs LMM)
hist(data$aci) # look at response variable
hist(log(data$aci))

# Note that scaled continuous values now have mean at 0 and variance = standard deviation = 1. If you wanted a prediction of acoustic activity at the mean temperature, I would predict at the mean value, i.e. temp = 0. Note that scaling changes our interpretation of betas.

# Should we omit the intercept, -1, assuming that activity goes to zero in response to extreme environmental covariates? Or is this unreasonable because those values are not present in the data, and even with the variable of time (season) there is no zero activity in the off-season.

glmm_global = glmer( # global model
  aci ~ stage + month + scale(tempmax) + scale(temp) + scale(windspeed) + scale(cloudcover) + (1|watershed),
  data = data, control = cntrl, family = Gamma(link='log'), nAGQ = 20
)

lmm_trans = lmer(
  log(aci) ~ stage + month + scale(tempmax) + scale(temp) + scale(windspeed) + scale(cloudcover) + (1|watershed),
  data = data
)

# Use "simulateResiduals" from DHARMa to do some diagnostics  
n = 1000
plot(simulateResiduals(fittedModel = glmm_global, n = n))
plot(simulateResiduals(fittedModel = lmm_trans, n = n))

check_err_identically_dist(glmm_global); title('GLMM')
check_err_identically_dist(lmm_trans); title('LMM log(y)')

#Deviance residuals
ypred = predict(glmm_global)
res = residuals(glmm_global, type = 'deviance')
plot(ypred,res)
hist(res)

check_err_norm_dist(glmm); title('GLMM')
check_err_norm_dist(lmm); title('LMM')
check_err_norm_dist(lmm_trans); title('LMM log(y)')
hist(residuals(glmm))
hist(residuals(lmm))
hist(residuals(lmm_trans))

qqnorm(unlist(ranef(lmm)$`watershed:site`), main = 'QQ plot (Random effect watershed:site)')
qqline(unlist(ranef(lmm)$`watershed:site`))

check_err_independent(glmm)
check_err_independent(lmm)
check_err_independent(lmm_trans)

# Look into deviance?

# Boxcox to identify transformation?

# Identify noisy outliers using diagnostics and points of inference for LMM
ggplot(data_raw, aes(x = date, y = aci, color = stage, shape = Noise)) + geom_point() + scale_color_manual(values = stage_colors) + labs(title = 'ACI per strata over time') # rainy observations have high ACI because of extreme amplitude fluctuations

# Note that with mixed models we cannot use the hat matrix to find outliers or points of influence, so we are left with looking at a subset of dimensions for outliers (e.g. normal boxplots)

# Remove noisy outliers and compare GLMM vs LMM again

# Develop a set of models to consider, ideally these have arisen from careful thought:
#  - They may represent different hypotheses about how our system works
#  - They may represent different combinations of variables that could be useful for prediction

# Do these alternative models fit reasonably well (i.e., do they meet the assumptions of the modeling procedure)?

# Model selection ##############################################################################

# 3. Keep random effects as is and fit different fixed effects (use AIC)

glmm_null = glmer(
  aci ~ (1|watershed:site), data,
  control = cntrl,
  family = Gamma(link='log'), # distribution of data y~f(y), and link function
  nAGQ = 1
) 
glmm = glmer(
  aci ~ stage + month + temperature + (1|watershed:site), data,
  control = cntrl,
  family = Gamma(link='log'),
  nAGQ = 1
)
lmm_null = lmer(
  log(aci) ~ (1|watershed:site), data
)
lmm = lmer(
  log(aci) ~ stage + month + temperature + (1|watershed:site), data
)

models = list(
  glmm_null, glmm, lmm_null, lmm
)

# Once you’ve decided on your random effects, proceed to model selection for fixed effects (but don’t use REML when you do)
# Evaluate goodness-of-fit
# F-tests to compare models vs AIC?
# We have three general approaches: out-of-sample procedures, within-sample procedures, and cross-validation

# Compare:
# Peak temperature, mean temperature, hour temperature
# Month, half season, third season
# Interaction between month and stage?

# Compare AIC
# NOTE: YOU CAN'T COMPARE AIC OF DIFFERENT MODEL STRUCTURES!!!
# Compare diagnostics instead!!
model_selection = t(data.frame(sapply(models, extractAIC)))
model_selection = cbind(model_selection,
                        sapply(models, function(m) { as.character(attributes(m)$call[1]) }))
model_selection = cbind(model_selection,
                        data.frame(sapply(models, function(m) { paste(attributes(m)$call[2], collapse=' ') })))
colnames(model_selection) = c('edf', 'AIC', 'type', 'model')
row.names(model_selection) = NULL
model_selection

# Inference ####################################################################################

# We can look at various outputs from the model 
summary(lmm_trans)
VarCorr(lmm_trans)
fixef(lmm_trans)
confint(lmm_trans, level = 0.95, method = 'profile', nsim = 500)

# Confidence intervals will generally require bootstrapping
# Coefficient interpretation

# Findings:
# - As expected, forest stage habitat has the greatest effect on activity.
# - Thinned stands do improve habitat! So much so that they more closely resemble the acoustic activity of mature forests than they do non-thinned stands of the same age. (Show in smoothed mean ACI by month graph too, with dotted line for thinned)
# - Interestingly, the mean activity of early-stage forests across the breeding season is nearly TODO times more than the other habitats
# - Peaks in activity occur during April and June, which correspond with known breeding schedules
# - The effect of amtospheric covariates are...

# Next steps:
# - Include additional covariates for elevation and aspect
# - Look at different times of day, besides dawn chorus
# - Look at species-specific results. Do these trends hold across species? How does species richness change?
