data_raw = read.csv('qerm-514/output/qerm514.csv')
data_raw$watershed = factor(data_raw$watershed)
data_raw$site = factor(data_raw$site)
data_raw$date = as.Date(data_raw$date)
data_raw$month = factor(month(data_raw$date))
data_raw$noise = factor(data_raw$noise)
data_raw$stage = factor(data_raw$stage, levels = c('Early', 'Mid', 'Mid (Thinned)', 'Late'))
data = data_raw[data_raw$noise!='rain', ]

optimizer = 'bobyqa'
optCtrl = list(maxfun=1000000)
# -------------------------------------------

model_glmer = glmer(
  aci ~ stage + month + scale(cloudcover) + scale(humidity) + scale(precip) + scale(temp) + scale(tempmax) + scale(windgust) + scale(windspeed) + (1|watershed:site),
  data = data,
  control = glmerControl(optimizer = optimizer, optCtrl = optCtrl),
  family = Gamma(link='log'), nAGQ = 1  # Laplace
)

model_glmmTMB = glmmTMB( # Laplace
  aci ~ stage + month + scale(cloudcover) + scale(humidity) + scale(precip) + scale(temp) + scale(tempmax) + scale(windgust) + scale(windspeed) + (1|watershed:site),
  data = data,
  control = glmmTMBControl(),
  family = Gamma(link='log')
)

model_lmer_lognormal = lmer(
  log(aci) ~ stage + month + scale(cloudcover) + scale(humidity) + scale(precip) + scale(temp) + scale(tempmax) + scale(windgust) + scale(windspeed) + (1|watershed:site),
  data = data
)

rbind(
  unlist(fixef(model_glmer))[1:7],
  unlist(fixef(model_glmmTMB))[1:7],
  unlist(fixef(model_lmer_lognormal))[1:7]
)

head(data.frame(
  fitted(model_glmer),
  fitted(model_glmmTMB),
  exp(fitted(model_lmer_lognormal))
))

head(data.frame(
  predict(model_glmer),
  predict(model_glmmTMB),
  predict(model_lmer_lognormal)
))

head(data.frame(
  residuals(model_glmer),
  residuals(model_glmmTMB),
  residuals(model_lmer_lognormal)
))

check_err_identically_dist = function(model) {
  plot(fitted(model), residuals(model),
       xlab = 'Fitted', ylab = 'Residuals',
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


# DHARMa diagnostics (using standardized residuals)
# See https://cran.r-project.org/web/packages/DHARMa/vignettes/DHARMa.html
sim_glmer = simulateResiduals(model_glmer)
sim_glmmTMB = simulateResiduals(model_glmmTMB)
sim_lmer = simulateResiduals(model_lmer_lognormal)

# QQ plot of residuals (expected vs observed)
par(mfrow = c(1, 3))
testUniformity(sim_glmer)
testUniformity(sim_glmmTMB)
testUniformity(sim_lmer)

check_err_norm_dist(model_glmer)

# Residuals vs predictions
par(mfrow = c(1, 3))
testQuantiles(sim_glmer)
testQuantiles(sim_glmmTMB)
testQuantiles(sim_lmer)

par(mfrow = c(1, 3))
check_err_identically_dist(model_glmer)
check_err_identically_dist(model_glmmTMB)
check_err_identically_dist(model_lmer_lognormal)

# Observed vs fitted values
par(mfrow = c(1, 3))
plot(data$aci, fitted(model_glmer), xlab = 'Observed', ylab = 'Fitted')
abline(a=0, b=1, col='red')
plot(data$aci, fitted(model_glmmTMB), xlab = 'Observed', ylab = 'Fitted')
abline(a=0, b=1, col='red')
plot(data$aci, exp(fitted(model_lmer_lognormal)), xlab = 'Observed', ylab = 'Fitted')
abline(a=0, b=1, col='red')

par(mfrow = c(3, 1))
testDispersion(sim_glmer)
testDispersion(sim_glmmTMB)
testDispersion(sim_lmer)

par(mfrow = c(3, 1))
testOutliers(sim_glmer)
testOutliers(sim_glmmTMB)
testOutliers(sim_lmer)

# Test for within-group uniformity and homogeneity of variance (Levene's)
par(mfrow = c(1, 2))
testCategorical(sim_glmer, catPred = data$stage)
testCategorical(sim_glmer, catPred = data$month)
testCategorical(sim_glmmTMB, catPred = data$stage)
testCategorical(sim_glmmTMB, catPred = data$month)
testCategorical(sim_lmer, catPred = data$stage)
testCategorical(sim_lmer, catPred = data$month)

# Temporal autocorrelation (Durbin-Watson), aggregating residuals by time
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
  recalculateResiduals(res_glmer, sel = site_to_test),
  time = unique(data[site_to_test, 'date'])
)

# TODO: testSpatialAutocorrelation





########

testData = data.frame(
  y = rgamma(n = 1000, shape = 3, scale = 2),
  
)

m_glmer = glmer(
  observedResponse ~ Environment1 + (1|group),
  family = Gamma(), data = testData
)
m_glmmTMB = glmmTMB(
  observedResponse ~ Environment1 + (1|group), 
  family = Gamma(), data = testData
)

s_glmer = simulateResiduals(m_glmer, plot = F)
s_glmmTMB = simulateResiduals(m_glmmTMB, plot = F)

plot(s_glmer)
plot(s_glmmTMB)

par(mfrow = c(1,2))
check_err_identically_dist(m_glmer)
check_err_identically_dist(m_glmmTMB)
