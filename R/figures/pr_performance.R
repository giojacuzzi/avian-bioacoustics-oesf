# Plot performance comparison between pre-trained and custom model

library(dplyr)
library(tools)
library(ggplot2)
library(cowplot)
library(patchwork)

path_pretrained = '/Users/giojacuzzi/Downloads/perf/data/validation/Custom/pre-trained'
path_custom = '/Users/giojacuzzi/Downloads/perf/data/validation/Custom/custom'

labels_to_plot = 'all'
labels_to_plot = c('marbled murrelet')

load_perf = function(path, model_tag) {
  files = list.files(path = path, pattern = "\\.csv$", full.names = TRUE)
  perf = lapply(files, function(file) {
    data = read.csv(file)
    data$label = file_path_sans_ext(basename(file))
    data$model = model_tag
    return(data)
  })
  perf = bind_rows(perf)
  return(perf)
}

perf_pretrained = load_perf(path_pretrained, 'pretrained')
perf_custom     = load_perf(path_custom, 'custom')

perf = bind_rows(perf_pretrained, perf_custom)
perf$label = factor(perf$label)
perf$model = factor(perf$model, levels = c('pretrained', 'custom'))

lines = ggplot(perf, aes(x = threshold)) +
  geom_line(aes(y = recall, linetype = "Recall", color = model)) +
  geom_line(aes(y = precision, linetype = "Precision", color = model)) +
  facet_wrap(~ label, scales = "free_y") +
  scale_color_manual(values = c("custom" = "royalblue", "pretrained" = "salmon")) +
  scale_linetype_manual(values = c("Recall" = "dotted", "Precision" = "solid")) +
  labs(x = "Threshold", y = "Value") +
  theme_minimal()
lines

lines = ggplot(perf, aes(x = recall, y = precision, color = model)) +
  geom_line() +
  geom_line(aes(color = model)) +
  facet_wrap(~ label, scales = "free_y") +
  scale_color_manual(values = c("custom" = "royalblue", "pretrained" = "salmon")) +
  labs(x = "Recall", y = "Precision") +
  theme_minimal()
lines

histograms = ggplot(perf, aes(x = threshold)) +
  geom_histogram(data = subset(perf, model == "pretrained"), fill = "red", alpha = 0.55, bins = 12) +
  geom_histogram(data = subset(perf, model == "custom"), fill = "blue", alpha = 0.55, bins = 12) +
  facet_wrap(~ label, scales = "free_y") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) +
  coord_cartesian(ylim = c(0, 10)) +
  labs(x = "Score Threshold", y = "Number of Detections") +
  theme_minimal()
histograms


if (labels_to_plot == 'all') {
  labels_to_plot = unique(perf$label)
}
plots_list <- list()

# Iterate over each label
for (l in labels_to_plot) {
  
  # Line plot
  line_plot <- ggplot(subset(perf, label == l), aes(x = threshold)) +
    geom_line(aes(y = recall, linetype = "Recall", color = model, alpha = 0.55)) +
    geom_line(aes(y = precision, linetype = "Precision", color = model, alpha = 0.55)) +
    scale_color_manual(values = c("pretrained" = "red", "custom" = "blue")) +
    scale_linetype_manual(values = c("Recall" = "dotted", "Precision" = "solid")) +
    scale_x_continuous(minor_breaks = NULL) +
    scale_y_continuous(minor_breaks = NULL) +
    labs(x = NULL, y = NULL) +
    ggtitle(l) +
    theme_minimal() +
    theme(legend.position = "none")

  # Histogram plot
  histogram_plot <- ggplot(subset(perf, label == l), aes(x = threshold)) +
    geom_histogram(data = subset(perf, label == l & model == "pretrained"), fill = "red", alpha = 0.55, bins = 12) +
    geom_histogram(data = subset(perf, label == l & model == "custom"), fill = "blue", alpha = 0.55, bins = 12) +
    scale_x_continuous(minor_breaks = NULL) +
    scale_y_continuous(minor_breaks = NULL) +
    coord_cartesian(ylim = c(0, 10)) +
    labs(x = NULL, y = NULL) +
    theme_minimal() +
    theme(legend.position = "none", axis.text.x = element_blank())
  
  # Combine line plot and histogram plot vertically
  combined_plot <- line_plot / histogram_plot +
    plot_layout(heights = c(3,1))

  # Add combined plot to the list
  plots_list[[l]] <- combined_plot
}

# Combine all plots horizontally
combined_plots <- wrap_plots(plots_list)

# Print combined plots
print(combined_plots)

