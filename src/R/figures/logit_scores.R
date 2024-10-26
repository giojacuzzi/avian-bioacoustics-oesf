library(ggplot2)

label = 'Golden-crowned Kinglet'#'Sooty Grouse'

model_stub = 'custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0'

path_target = paste0('data/test/raw_predictions/', model_stub)
path_source = 'data/test/raw_predictions/pretrained'

files_target = list.files(path = path_target, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
nfiles_target = length(files_target)
files_source = list.files(path = path_source, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
nfiles_source = length(files_source)

max_count = nfiles_source

# Load predictions from source model
predictions_source = data.frame()
counter = 0
for (file in files_source) {
  counter = counter + 1
  print(paste(counter, '/', nfiles_source, ':', file))
  
  predictions = read.csv(file)
  predictions = predictions[predictions$common_name == label, ]
  
  if (!is.null(predictions) && is.data.frame(predictions)) {
    if (nrow(predictions) > 0) {
      predictions$model = 'Source'
      predictions_source = rbind(predictions_source, predictions)
    }
  }
  
  if (counter == max_count) {
    break
  }
}

# Load predictions from target model
predictions_target = data.frame()
counter = 0
for (file in files_target) {
  counter = counter + 1
  print(paste(counter, '/', nfiles_target, ':', file))
  
  predictions = read.csv(file)
  predictions = predictions[predictions$common_name == label, ]
  
  if (!is.null(predictions) && is.data.frame(predictions)) {
    if (nrow(predictions) > 0) {
      predictions$model = 'Target'
      predictions_target = rbind(predictions_target, predictions)
    }
  }
  
  if (counter == max_count) {
    break
  }
}

ylimit = 1000

theme_set(theme_light())

predictions = rbind(predictions_source, predictions_target)

ggplot(predictions, aes(x=logit, color=model, fill=model)) + scale_x_continuous(limits = c(-15, 15)) + geom_density(alpha=0.5)
ggplot(predictions, aes(x=logit, fill=model)) + scale_x_continuous(limits = c(-15, 15)) + geom_histogram(alpha=0.75,bins=1000, position='identity')
ggplot(predictions, aes(x=confidence, fill=model)) + coord_cartesian(ylim=c(0,2000)) + geom_histogram(alpha=0.75, bins=100, position='identity')
ggplot(predictions, aes(x=confidence, fill=model)) + scale_y_continuous(trans='log10') + geom_histogram(alpha=0.75, bins=100, position='identity')
ggplot(predictions, aes(x=confidence, colour=model)) + scale_y_continuous(trans='log10') + geom_density()

## Individual plots

ggplot(predictions_source, aes(x=logit)) + scale_x_continuous(limits = c(-15, 15)) + geom_density()
ggplot(predictions_target, aes(x=logit)) + scale_x_continuous(limits = c(-15, 15)) + geom_density()

ggplot(predictions_source, aes(x=logit)) + scale_x_continuous(limits = c(-15, 15)) + geom_histogram(bins=1000)
ggplot(predictions_target, aes(x=logit)) + scale_x_continuous(limits = c(-15, 15)) + geom_histogram(bins=1000)

ggplot(predictions_source, aes(x=confidence)) + scale_y_continuous(trans='log10') + geom_histogram(bins=100)
ggplot(predictions_target, aes(x=confidence)) + scale_y_continuous(trans='log10') + geom_histogram(bins=100)

ggplot(predictions_source, aes(x=confidence)) + scale_y_continuous(trans='log10') + geom_density()
ggplot(predictions_target, aes(x=confidence)) + scale_y_continuous(trans='log10') + geom_density()
