files_results = list.files(path = 'data/results/', pattern = "\\metrics_custom.csv$", full.names = TRUE, recursive = TRUE)

results = data.frame()
for (file in files_results) {
  data = read.csv(file)
  data$label = tolower(data$label)
  results = rbind(results, data)
}
results$dir = gsub("data/test/|/custom", "", results$model)

files_class_counts = unique(file.path('data/models/custom', paste0(results$dir, '/trained_class_counts.csv')))
class_counts = data.frame()
for (file in files_class_counts) {
  data = read.csv(file)
  names(data)[names(data) == "labels"] <- "label"
  data$label = tolower(sub(".*_", "", data$label))
  data$dir = basename(dirname(file))
  class_counts = rbind(class_counts, data)
}

merged = merge(results, class_counts, by = c("label", "dir"), all.x = TRUE)
merged = merged[!merged$label %in% c('abiotic aircraft', 'abiotic logging', 'abiotic rain', 'abiotic vehicle', 'abiotic wind', 'biotic anuran', 'biotic insect'),]

avg_line_data <- merged %>%
  group_by(count) %>%
  summarize(avg_PR_AUC = mean(PR_AUC, na.rm = TRUE), .groups = "drop")
approx_data <- data.frame(approx(avg_line_data$count, avg_line_data$avg_PR_AUC, n = 100))
colnames(approx_data) <- c("count", "avg_PR_AUC")

library(dplyr)
library(ggplot2)
for (l in unique(merged$label)) {
  print(l)
  data_l = merged[merged$label == l,]
  # data_l = merged
  p = ggplot() + # merged[merged$label == "pileated woodpecker",]
    geom_line(data = data_l, aes(x = count, y = PR_AUC, color = label, group = label), alpha = 0.5) +
    geom_point(data = data_l, aes(x = count, y = PR_AUC, color = label, group = label), alpha = 0.5) +
    # geom_text(data = merged %>%
    #             group_by(label) %>%
    #             summarize(count = last(count), PR_AUC = last(PR_AUC)),
    #           aes(x = count, y = PR_AUC, label = label, color = label),
    #           hjust = -0.1, vjust = 0, size = 3.5) +
    # geom_line(data = approx_data, aes(x = count, y = avg_PR_AUC), color = "black", size = 1, linetype = "dashed") +
    # coord_cartesian(xlim=c(0,75), ylim=c(0.0,1.0)) +
    # stat_smooth(data = merged, aes(x = count, y = PR_AUC, color = label, group = label), method="lm",formula=y~log(x), fill = NA) +
    # stat_smooth(data = merged, aes(x = count, y = PR_AUC), method="lm",formula=y~log(x)) +
    # stat_smooth(fill = NA) +
    labs(x = "Training samples", y = "PR AUC", title = l) +
    theme_minimal(); print(p)
  readline()
}

library(viridis)
plot_sample_size = ggplot(data = merged) +
  geom_line(aes(x = count, y = PR_AUC, color = count)) +
  scale_colour_viridis(option = "D") +
  facet_wrap(~ str_to_title(label), ncol = 7, scales = "free_x") +
  theme_bw() + theme(aspect.ratio = 1) +
  labs(color = "Samples", x = "PR AUC", y = "Training sample size")
plot_sample_size
ggsave(file=paste0("data/figures/plot_sample_size", ".png"), plot=plot_sample_size, width=12, height=16)

df_sorted <- merged %>% arrange(count)
model <- lm(PR_AUC ~ log(count), data = df_sorted)
plot(df_sorted$count, df_sorted$PR_AUC, main = "Logarithmic Regression Plot", xlab = "x_column", ylab = "y_column", pch = 19)
lines(df_sorted$count, predict(model), col = "blue", lwd = 2)
plot(model$residuals, main = "Residuals of Logarithmic Regression", ylab = "Residuals", xlab = "Fitted Values", pch = 19)
abline(h = 0, col = "red")

# Find asymptotes
asymptotes <- merged %>%
  group_by(label) %>%
  filter(PR_AUC == max(PR_AUC)) %>%
  slice_min(count) %>%
  ungroup()
print(asymptotes)

mean(class_counts[class_counts$dir == 'custom_S1_N5_LR0.001_BS5_HU0_LSFalse_US0_I0', 'count'])
mean(merged[merged$dir == 'custom_S1_N5_LR0.001_BS5_HU0_LSFalse_US0_I0', 'PR_AUC'])
