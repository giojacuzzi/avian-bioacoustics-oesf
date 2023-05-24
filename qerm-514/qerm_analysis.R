source('global.R')
library(ggplot2)
theme_set(theme_minimal())

# Get data from 2020, 2021
data = get_joined_data()

data = na.omit(data[data$DataYear %in% c(2021), ])

# Plot temperature data
ggplot(data, aes(x = SurveyDate, y = TempMax, color = Strata)) +
  geom_point() + labs(title = 'Temperature per strata')

# Temperature stats by strata
sort(tapply(data$TempMax, data$Strata, median))
sort(tapply(data$TempMax, data$Strata, mean))

# Get index data
data_acidx = read.csv('acoustic_indices/output/6am/results.csv')
data_acidx$SurveyDate = as.Date(data_acidx$SurveyDate)
data_acidx$SerialNo = factor(data_acidx$SerialNo)
data_acidx = left_join(data_acidx,
                       distinct(data[, c('WatershedID', 'StationName_AGG',
                                         'SurveyDate', 'SerialNo', 'Strata', 'TempMax')])
                       , by = c('SurveyDate', 'SerialNo'))
data_acidx$Month = factor(month(data_acidx$SurveyDate))
data_acidx$Year = factor(year(data_acidx$SurveyDate))
data_acidx = data_acidx[data_acidx$Year==2021,]
data_acidx = na.omit(data_acidx)

data_noise = read.csv('qerm-514/output/annotations.csv')
data_noise$Noise = factor(data_noise$Noise)
data_noise$SurveyDate = as.Date(data_noise$SurveyDate)
data_acidx = left_join(data_acidx,
                       distinct(data_noise[, c('Noise','SurveyDate', 'SerialNo', 'Note')])
                       , by = c('SurveyDate', 'SerialNo'))

## Visualize data

# Date and strata vs acdix
ggplot(data_acidx, aes(x = SurveyDate, y = ACI, color = Strata, shape = Noise)) +
  geom_point() + labs(title = 'ACI per strata over time')
ggplot(data_acidx, aes(x = SurveyDate, y = BIO, color = Strata, shape = Noise)) +
  geom_point() + labs(title = 'BIO per strata over time')

# Remove rain dates
data_acidx = data_acidx[!data_acidx$Noise=='rain' & !data_acidx$Noise=='wind', ]

# Strata vs acidx
ggplot(data_acidx, aes(x = Strata, y = ACI, color = Strata)) +
  geom_boxplot() + labs(title = 'ACI x strata')
ggplot(data_acidx, aes(x = Strata, y = BIO, color = Strata)) +
  geom_boxplot() + labs(title = 'BIO x strata')

# Temp vs acidx
ggplot(data_acidx, aes(x = TempMax, y = ACI, color = Strata)) +
  geom_point() + labs(title = 'ACI x temp per strata') + geom_smooth(method='lm')
ggplot(data_acidx, aes(x = TempMax, y = BIO, color = Strata)) +
  geom_point() + labs(title = 'BIO x temp per strata') + geom_smooth(method='lm')

# Get stats by strata
sort(tapply(data_acidx$ACI, data_acidx$Strata, median))
sort(tapply(data_acidx$BIO, data_acidx$Strata, median))

# Distribution of response variables
hist(data_acidx$ACI)
hist(log(data_acidx$ACI))
hist(data_acidx$BIO)
hist(log(data_acidx$BIO))

hist(data_acidx$TempMax)

####################################################################

library(lme4)
library(faraway)

# Fit a linear mixed model
model = lmer(ACI ~ (1|WatershedID:StationName_AGG) # log(BIO)?
             + Strata + TempMax + TempMax*Strata + Month, data_acidx)
# model = lmer(TempMax ~ (1|WatershedID:StationName_AGG) + Strata + Month, data_acidx)

# Plot the residuals against the fitted values
# plot(model)
plot(fitted(model), residuals(model),
     xlab = "Fitted Values", ylab = "Residuals",
     ylim = c(-max(residuals(model))-1,max(residuals(model))+1), pch = 19)
abline(h=0, col='blue')

model.test = lm(I(sqrt(abs(residuals(model)))) ~ I(fitted(model)))
sumary(model.test)
abline(model.test, col='red') # Plot fitted slope

# Are residuals normally distributed?
hist(residuals(model))
qqnorm(residuals(model))
qqline(residuals(model))

# Fit a generalized linear mixed model
model = glmer(ACI ~ (1|WatershedID:StationName_AGG)
             + Strata + TempMax + TempMax*Strata + Month, family=Gamma('log'),nAGQ=0, data=data_acidx)

# model = glmer(TempMax ~ (1|WatershedID:StationName_AGG)
              # + Strata + Month, family=Gamma('log'),nAGQ=0, data=data_acidx)

#####

library(car)
group = rep(0,nrow(data_acidx))
group[which(residuals(model)>median(residuals(model)))] <- 1
group = as.factor(group)
leveneTest(residuals(model), group)

# Large leverages?
th = 2*(length(coef(model))/nrow(data_acidx))
large.hats = which(hatvalues(model)>th)
halfnorm(hatvalues(model))
hatvalues(model)[large.hats]
data_acidx[large.hats,]

# Outliers? 
studentized = rstudent(model)
halfnorm(studentized)
outlierTest(model)

# Influential observations?
cook = cooks.distance(model)
halfnorm(cook, 2, ylab="Cook’s Distance")

