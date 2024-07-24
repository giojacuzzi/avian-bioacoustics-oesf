# Plot species performance

library(dplyr)
library(tools)
library(ggplot2)
library(cowplot)
library(patchwork)

path = '/Users/giojacuzzi/repos/avian-bioacoustics-oesf/data/cache/test_validate_and_evaluate_perf'

labels_to_plot = 'all'
labels_to_plot = c('marbled murrelet')
label_to_plot = 'pacific-slope flycatcher'

load_perf = function(path, model_tag='') {
  files = list.files(path = path, pattern = "\\.csv$", full.names = TRUE)
  perf = lapply(files, function(file) {
    data = read.csv(file)
    data$label = file_path_sans_ext(basename(file))
    if (model_tag != '') {
      data$model = model_tag
    }
    return(data)
  })
  perf = bind_rows(perf)
  return(perf)
}

perf = load_perf(path)

label_perf = perf[perf['label'] == label_to_plot, ]
label_perf = rbind(data.frame(threshold = 0.0, precision = 0.0, recall = 1.0, label = label_to_plot), label_perf)
label_perf = rbind(label_perf, data.frame(threshold = 1.0, precision = 1.0, recall = 0.0, label = label_to_plot))

# Threshold precision-recall plot
plot_threshold_pr <- ggplot(label_perf, aes(x = threshold)) +
  geom_path(aes(y = recall, linetype = "solid"), color = 'royalblue', alpha = 0.8) +
  geom_path(aes(y = precision, linetype = "solid"), color = 'salmon', alpha = 0.8) +
  # scale_linetype_manual(values = c("Recall" = "dotted", "Precision" = "solid")) +
  scale_x_continuous(minor_breaks = NULL) +
  scale_y_continuous(minor_breaks = NULL) +
  coord_fixed(ratio = 1) +
  labs(x = "Threshold", y = "Performance") +
  ggtitle(label_to_plot) +
  theme_minimal() +
  theme(legend.position = "none")
plot_threshold_pr

# Precision-recall plot
plot_pr = ggplot(label_perf, aes(x = recall, y = precision)) +
  geom_path(alpha = 0.8, color = 'black') +
  # facet_wrap(~ label, scales = "free_y") +
  # scale_color_manual(values = c("custom" = "royalblue", "pretrained" = "salmon")) +
  labs(x = "Recall", y = "Precision") +
  coord_fixed(ratio = 1) +
  ggtitle(label_to_plot) +
  theme_minimal()
plot_pr

# Histogram plot
plot_histogram <- ggplot(label_perf, aes(x = threshold)) +
  geom_histogram(data = label_perf, fill = "red", alpha = 0.55, bins = 12) +
  scale_x_continuous(minor_breaks = NULL) +
  scale_y_continuous(minor_breaks = NULL) +
  coord_cartesian(ylim = c(0, 10)) +
  labs(x = NULL, y = NULL) +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_blank())
plot_histogram

lines = ggplot(label_perf, aes(x = threshold)) +
  geom_path(aes(y = recall, linetype = "Recall", color = 'black')) +
  geom_path(aes(y = precision, linetype = "Precision", color = 'black')) +
  facet_wrap(~ label, scales = "free_y") +
  # scale_color_manual(values = c("custom" = "royalblue", "pretrained" = "salmon")) +
  scale_linetype_manual(values = c("Recall" = "dotted", "Precision" = "solid")) +
  labs(x = "Threshold", y = "Value") +
  theme_minimal()
lines

lines = ggplot(label_perf, aes(x = recall, y = precision, color = 'black')) +
  geom_path() +
  facet_wrap(~ label, scales = "free_y") +
  scale_color_manual(values = c("custom" = "royalblue", "pretrained" = "salmon")) +
  labs(x = "Recall", y = "Precision") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal()
lines

