# Plot performance comparison between pre-trained and custom model

library(dplyr)
library(tools)
library(ggplot2)
library(cowplot)
library(patchwork)
library(stringr)
source('src/R/figures/global.R')

custom_model_stub = 'custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0'
# /Users/giojacuzzi/repos/avian-bioacoustics-oesf/data/test/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/custom/threshold_perf
path_pretrained = paste('/Users/giojacuzzi/repos/avian-bioacoustics-oesf/data/test/', custom_model_stub, '/pre-trained/threshold_perf', sep='')
path_custom = paste('/Users/giojacuzzi/repos/avian-bioacoustics-oesf/data/test/', custom_model_stub, '/custom/threshold_perf', sep='')

labels_to_plot = 'all'
labels_to_plot = c('marbled murrelet', 'pacific-slope flycatcher')

class_labels = read.csv('/Users/giojacuzzi/repos/avian-bioacoustics-oesf/data/class_labels.csv')

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
perf_pretrained$f1 = 2*perf_pretrained$recall * perf_pretrained$precision/(perf_pretrained$recall+perf_pretrained$precision)

perf_custom     = load_perf(path_custom, 'custom')
perf_custom$f1 = 2*perf_custom$recall * perf_custom$precision/(perf_custom$recall+perf_custom$precision)

perf = bind_rows(perf_pretrained, perf_custom)
# perf = perf %>% filter(!str_detect(label, paste(c(labels_to_remove, class_labels[class_labels$train == 0, 'label']), collapse = "|")))
# TODO: Filter out species classes that were not present in the study area
perf = perf %>% filter(!str_detect(label, paste(c(labels_to_remove), collapse = "|")))
perf$label = factor(str_to_title(perf$label))
perf$model = factor(perf$model, levels = c('pretrained', 'custom'))
perf$model = recode(perf$model, "pretrained" = "Source", "custom" = "Target")

# Figure: Threshold performance for all classes
plot_threshold_pr = ggplot(perf, aes(x = threshold)) +
  geom_path(aes(y = recall, linetype = "Recall", color = model)) +
  geom_path(aes(y = precision, linetype = "Precision", color = model)) +
  geom_path(aes(y = f1, linetype = "F1", color = model)) +
  facet_wrap(~ label, ncol = 4, scales = "free_y") +
  scale_color_manual(values = c("Target" = "royalblue", "Source" = "salmon")) +
  scale_linetype_manual(values = c("Recall" = "dashed", "Precision" = "solid", "F1" = "dotted")) +
  labs(title = "Sample level test performance", x = "Threshold", y = "Performance", color = 'Model', linetype = 'Metric') +
  theme_minimal()
plot_threshold_pr

# Figure: Precision-Recall AUC for all classes
plot_pr = ggplot(perf, aes(x = recall, y = precision, color = model)) +
  geom_path() +
  geom_path(aes(color = model)) +
  facet_wrap(~ label, ncol = 4, scales = "free_y") +
  scale_color_manual(values = c("Target" = "royalblue", "Source" = "salmon")) +
  labs(x = "Recall", y = "Precision") +
  theme_minimal()
plot_pr

plot_histogram = ggplot(perf, aes(x = threshold)) +
  geom_histogram(data = subset(perf, model == "Source"), fill = "red", alpha = 0.55, bins = 12) +
  geom_histogram(data = subset(perf, model == "Target"), fill = "blue", alpha = 0.55, bins = 12) +
  facet_wrap(~ label, scales = "free_y") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) +
  coord_cartesian(ylim = c(0, 40)) +
  labs(x = "Score Threshold", y = "Number of Detections") +
  theme_minimal()
plot_histogram

# plot_histogram = ggplot(perf, aes(x = threshold)) +
#   geom_density(data = subset(perf, model == "Source"), fill = "red", alpha = 0.55) +
#   geom_density(data = subset(perf, model == "Target"), fill = "blue", alpha = 0.55) +
#   facet_wrap(~ label, scales = "free_y") +
#   scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) +
#   coord_cartesian(ylim = c(0, 40)) +
#   labs(x = "Score Threshold", y = "Number of Detections") +
#   theme_minimal()
# plot_histogram


# library(diptest)
# library(LaplacesDemon)
# label_to_test = 'song sparrow'
# pretrained_dist = subset(subset(perf, model == 'pretrained'), label == label_to_test, select = threshold)
# custom_dist = subset(subset(perf, model == 'custom'), label == label_to_test, select = threshold)
# plot(pretrained_dist)
# plot(custom_dist)
# is.unimodal(unlist(pretrained_dist))
# is.unimodal(unlist(custom_dist))
# is.multimodal(unlist(pretrained_dist))
# is.multimodal(unlist(custom_dist))
# dip.test(unlist(pretrained_dist))
# dip.test(unlist(custom_dist))



if (labels_to_plot == 'all') {
  labels_to_plot = unique(perf$label)
}


# Plot specified labels independently
for (l in str_to_title(labels_to_plot)) {
  
  plots_list <- list()
  
  # Threshold precision-recall plot
  plot_threshold_pr <- ggplot(subset(perf, label == l), aes(x = threshold)) +
    geom_path(aes(y = recall, linetype = "Recall", color = model, alpha = 0.55)) +
    geom_path(aes(y = precision, linetype = "Precision", color = model, alpha = 0.55)) +
    scale_color_manual(values = c("Source" = "red", "Target" = "blue")) +
    scale_linetype_manual(values = c("Recall" = "dotted", "Precision" = "solid")) +
    scale_x_continuous(minor_breaks = NULL) +
    scale_y_continuous(minor_breaks = NULL) +
    coord_fixed(ratio = 1) +
    labs(x = "Threshold", y = "Performance") +
    ggtitle(l) +
    theme_minimal() +
    theme(legend.position = "none")
  plot_threshold_pr
  
  # Precision-recall plot
  plot_pr = ggplot(subset(perf, label == l), aes(x = recall, y = precision, color = model)) +
    geom_path() +
    geom_path(aes(color = model)) +
    # facet_wrap(~ label, scales = "free_y") +
    scale_color_manual(values = c("Target" = "royalblue", "Source" = "salmon")) +
    labs(x = "Recall", y = "Precision") +
    coord_fixed(ratio = 1) +
    theme_minimal()
  plot_pr

  # Histogram plot
  plot_histogram <- ggplot(subset(perf, label == l), aes(x = threshold)) +
    geom_histogram(data = subset(perf, label == l & model == "Source"), fill = "red", alpha = 0.55, bins = 12) +
    geom_histogram(data = subset(perf, label == l & model == "Target"), fill = "blue", alpha = 0.55, bins = 12) +
    scale_x_continuous(minor_breaks = NULL) +
    scale_y_continuous(minor_breaks = NULL) +
    coord_cartesian(ylim = c(0, 10)) +
    labs(x = NULL, y = NULL) +
    theme_minimal() +
    theme(legend.position = "none", axis.text.x = element_blank())
  plot_histogram
  
  # plot_density <- ggplot(subset(perf, label == l), aes(x = threshold)) +
  #   geom_density(data = subset(perf, label == l & model == "Source"), fill = "red", alpha = 0.55, adjust = 10) +
  #   geom_density(data = subset(perf, label == l & model == "Target"), fill = "blue", alpha = 0.55, adjust = 10) +
  #   scale_x_continuous(minor_breaks = NULL) +
  #   scale_y_continuous(minor_breaks = NULL) +
  #   coord_cartesian(ylim = c(0, 10)) +
  #   labs(x = NULL, y = NULL) +
  #   theme_minimal() +
  #   theme(legend.position = "none", axis.text.x = element_blank())
  # plot_density
  
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

