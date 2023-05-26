# Gio Jacuzzi - QERM 514 final project

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
  hist(residuals(model))
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
data_raw = left_join(
  data_raw,
  distinct(data_atmo[, c('SurveyDate', 'temp','humidity', 'precip', 'windgust', 'windspeed', 'cloudcover', 'conditions')]), by = c('SurveyDate'))

# Factor breeding season
data_raw$season_halves = cut(data_raw$SurveyDate, 2, labels=c('Early', 'Late'))
data_raw$season_thirds = cut(data_raw$SurveyDate, 3, labels=c('Early', 'Mid', 'Late'))

# Scale ACI
data_raw$aci = data_raw$ACI - min(data_raw$ACI) + 1 # TODO: scale to noise floor ACI == 0

# Rename columns
colnames(data_raw)[colnames(data_raw) == 'WatershedID'] = 'watershed'
colnames(data_raw)[colnames(data_raw) == 'StationName_AGG'] = 'site'
colnames(data_raw)[colnames(data_raw) == 'SurveyDate'] = 'date'
colnames(data_raw)[colnames(data_raw) == 'Strata'] = 'stage'
colnames(data_raw)[colnames(data_raw) == 'Month'] = 'month'
colnames(data_raw)[colnames(data_raw) == 'TempMax'] = 'temperature'

# Reorder stage factor
levels(data_raw$stage)[levels(data_raw$stage) == 'STAND INIT'] = 'Early'
levels(data_raw$stage)[levels(data_raw$stage) == 'COMP EXCL'] = 'Mid'
levels(data_raw$stage)[levels(data_raw$stage) == 'THINNED'] = 'Mid (Thinned)'
levels(data_raw$stage)[levels(data_raw$stage) == 'MATURE'] = 'Late'
data_raw$stage = factor(data_raw$stage, levels = c('Early', 'Mid', 'Mid (Thinned)', 'Late'))
stage_colors = c('#A25B5B', '#A4BE7B', '#54436B', '#285430')

# Create dataset without noisy dates
data = data_raw[data_raw$Noise=='', ] # [rain, wind, other]
data_noisy = data_raw

# Visualize data ###############################################################################
theme_set(theme_minimal())

ggplot(data_raw, aes(x = date, y = aci, color = stage)) + geom_point() +
  scale_color_manual(values = stage_colors) + labs(title = 'ACI vs strata over the breeding season')

ggplot(data_raw, aes(x = season_thirds, y = aci)) + geom_boxplot() + labs(title = 'ACI vs season')

ggplot(data_raw, aes(x = stage, y = aci, fill = stage)) + geom_boxplot() + scale_fill_manual(values = stage_colors) + labs(title = 'ACI vs stage')

ggplot(data_raw, aes(x = date, y = temperature, color = stage)) + geom_point() + scale_color_manual(values = stage_colors) + labs(title = 'Temperature vs strata')

# Hypothesis ###################################################################################

# TODO: songbird activity
# Make your assumptions on the data clear, particularly for the acidx
# Hypothesis "I think the world works like this" > "If that's true, I expect to see this in the data"

# A good analysis begins with a good question A good question is... An interesting question: it’s a question that relates to something that is relevant and important to basic science or management science A question with an informative answer. By answering the question, we will be able to distinguish between hypotheses or in some other way provide critical information to address the interesting scientific issue.

# Model design #################################################################################

# Start with random effects. Think hard about your problem: are there some likely grouping factors in your data? Identify important grouping factors based on bootstrap LRT (or just leave them all in...)

# We can follow these steps:
# 1. Fit a model with all of the possible fixed effects included
# 2. Keep the fixed effects constant and search for random effects (use bootstrap LRT)

# Once you’ve decided on your random effects, proceed to model selection for fixed effects (but don’t use REML when you do). How to structure the design matrix?

   # e.g. (1|pond:month) -> we estimate a random effect for each month at each pond - the effect of month can vary by pond

# Develop a full model based on the distribution of your data (GLMM vs LMM)

# Boxcox to identify transformation?

glmm_null = glmer(
  aci ~ (1|watershed:site), data,
  family = Gamma(link='log'), # distribution of data y~f(y), and link function
  nAGQ = 1
) 
glmm = glmer(
  aci ~ stage + month + temperature + (1|watershed:site), data,
  family = Gamma(link='log'),
  nAGQ = 0 # only way to get convergence (less accurate); 1 is Laplace, >1 Gauss-Hermite
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

# Identify noisy outliers using diagnostics and points of inference for LMM
ggplot(data_raw, aes(x = date, y = aci, color = stage, shape = Noise)) + geom_point() + scale_color_manual(values = stage_colors) + labs(title = 'ACI per strata over time') # rainy observations have high ACI because of extreme amplitude fluctuations

# Remove noisy outliers and compare GLMM vs LMM again

# Develop a set of models to consider, ideally these have arisen from careful thought:
#  - They may represent different hypotheses about how our system works
#  - They may represent different combinations of variables that could be useful for prediction

# Do these alternative models fit reasonably well (i.e., do they meet the assumptions of the modeling procedure)?

# Model selection ##############################################################################

# 3. Keep random effects as is and fit different fixed effects (use AIC)

# Once you’ve decided on your random effects, proceed to model selection for fixed effects (but don’t use REML when you do)
# Evaluate goodness-of-fit
# F-tests to compare models vs AIC?
# We have three general approaches: out-of-sample procedures, within-sample procedures, and cross-validation

# Peak temperature, mean temperature, hour temperature
# Month, half season, third season

summary(lmm)
check_err_identically_dist(lmm)
check_err_norm_dist(lmm)
check_err_independent(lmm)

# Compare AIC
model_selection = t(data.frame(sapply(models, extractAIC)))
model_selection = cbind(model_selection,
                        sapply(models, function(m) { as.character(attributes(m)$call[1]) }))
model_selection = cbind(model_selection,
                        data.frame(sapply(models, function(m) { paste(attributes(m)$call[2], collapse=' ') })))
colnames(model_selection) = c('edf', 'AIC', 'type', 'model')
row.names(model_selection) = NULL
model_selection

# Inference ####################################################################################

# Confidence intervals will generally require bootstrapping

# Coefficient interpretation
