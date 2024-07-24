# Plot performance comparison between pre-trained and custom model

library(dplyr)
library(tools)
library(ggplot2)
library(cowplot)
library(patchwork)

path_pretrained = '/Users/giojacuzzi/Downloads/perf/data/validation/Custom/pre-trained'
path_custom = '/Users/giojacuzzi/Downloads/perf/data/validation/Custom/custom'

labels_to_plot = 'all'
labels_to_plot = c('marbled murrelet', 'pacific-slope flycatcher')

load_perf = function(path, model_tag) {
  files = list.files(path = path, pattern = "\\.csv$", full.names = TRUE)
  perf = lapply(files, function(file) {
    label = file_path_sans_ext(basename(file))
    data = read.csv(file)
    data$label = label
    data$model = model_tag
    # Add missing values
    data = rbind(data.frame(threshold = 0.0, precision = 0.0, recall = 1.0, label = label, model = model_tag), data)
    data = rbind(data, data.frame(threshold = 1.0, precision = 1.0, recall = 0.0, label = label, model = model_tag))
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

plot_threshold_pr = ggplot(perf, aes(x = threshold)) +
  geom_path(aes(y = recall, linetype = "Recall", color = model)) +
  geom_path(aes(y = precision, linetype = "Precision", color = model)) +
  facet_wrap(~ label, scales = "free_y") +
  scale_color_manual(values = c("custom" = "royalblue", "pretrained" = "salmon")) +
  scale_linetype_manual(values = c("Recall" = "dotted", "Precision" = "solid")) +
  labs(x = "Threshold", y = "Value") +
  theme_minimal()
plot_threshold_pr

plot_pr = ggplot(perf, aes(x = recall, y = precision, color = model)) +
  geom_path() +
  geom_path(aes(color = model)) +
  facet_wrap(~ label, scales = "free_y") +
  scale_color_manual(values = c("custom" = "royalblue", "pretrained" = "salmon")) +
  labs(x = "Recall", y = "Precision") +
  theme_minimal()
plot_pr

plot_histogram = ggplot(perf, aes(x = threshold)) +
  geom_histogram(data = subset(perf, model == "pretrained"), fill = "red", alpha = 0.55, bins = 12) +
  geom_histogram(data = subset(perf, model == "custom"), fill = "blue", alpha = 0.55, bins = 12) +
  facet_wrap(~ label, scales = "free_y") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) +
  coord_cartesian(ylim = c(0, 10)) +
  labs(x = "Score Threshold", y = "Number of Detections") +
  theme_minimal()
plot_histogram


if (labels_to_plot == 'all') {
  labels_to_plot = unique(perf$label)
}


# Plot specified labels independently
for (l in labels_to_plot) {
  
  plots_list <- list()
  
  # Threshold precision-recall plot
  plot_threshold_pr <- ggplot(subset(perf, label == l), aes(x = threshold)) +
    geom_path(aes(y = recall, linetype = "Recall", color = model, alpha = 0.55)) +
    geom_path(aes(y = precision, linetype = "Precision", color = model, alpha = 0.55)) +
    scale_color_manual(values = c("pretrained" = "red", "custom" = "blue")) +
    scale_linetype_manual(values = c("Recall" = "dotted", "Precision" = "solid")) +
    scale_x_continuous(minor_breaks = NULL) +
    scale_y_continuous(minor_breaks = NULL) +
    coord_fixed(ratio = 1) +
    labs(x = "Threshold", y = "Performance") +
    ggtitle(l) +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Precision-recall plot
  plot_pr = ggplot(subset(perf, label == l), aes(x = recall, y = precision, color = model)) +
    geom_path() +
    geom_path(aes(color = model)) +
    # facet_wrap(~ label, scales = "free_y") +
    scale_color_manual(values = c("custom" = "royalblue", "pretrained" = "salmon")) +
    labs(x = "Recall", y = "Precision") +
    coord_fixed(ratio = 1) +
    theme_minimal()

  # Histogram plot
  plot_histogram <- ggplot(subset(perf, label == l), aes(x = threshold)) +
    geom_histogram(data = subset(perf, label == l & model == "pretrained"), fill = "red", alpha = 0.55, bins = 12) +
    geom_histogram(data = subset(perf, label == l & model == "custom"), fill = "blue", alpha = 0.55, bins = 12) +
    scale_x_continuous(minor_breaks = NULL) +
    scale_y_continuous(minor_breaks = NULL) +
    coord_cartesian(ylim = c(0, 10)) +
    labs(x = NULL, y = NULL) +
    theme_minimal() +
    theme(legend.position = "none", axis.text.x = element_blank())
  
  # Combine line plot and histogram plot vertically
  combined_plot <- plot_threshold_pr + plot_pr + plot_histogram
    # plot_layout(ncol = 2, heights = c(3,1))

  # Add combined plot to the list
  plots_list[[l]] <- combined_plot
  
  # Combine all plots horizontally
  combined_plots <- wrap_plots(plots_list)
  
  # Print combined plots
  print(combined_plots)
}

