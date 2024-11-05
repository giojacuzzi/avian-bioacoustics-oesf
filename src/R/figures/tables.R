
# Table S3 - Sample performance
{
  source = read.csv('data/results/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/sample_perf/metrics_pre-trained.csv')
  target = read.csv('data/results/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/sample_perf/metrics_custom.csv')
  
  data = rbind(target, source)
  data[data$model == 'data/test/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/pre-trained', 'model'] = 'source'
  data[data$model == 'data/test/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/custom', 'model'] = 'target'
  data$label = tolower(data$label)
  
  data = data[, !names(data) %in% c("conf_max", "p_Tp", "p_T095", "r_T095", "p_T09", "r_T09", "p_T05", "r_T05", "N", "N_unk", "class_ratio")]
  data = data[order(data$model, data$label), ]
  
  excluded_and_not_present_species_labels = c(unique(data[data$N_pos == 0, 'label']), "american crow", "american goldfinch", "bald eagle", "macgillivray’s warbler", "northern goshawk", "red-tailed hawk", "sharp-shinned hawk", "yellow warbler")
  
  # Among the 48 bird species present...
  summary(data[data$model == "source" & !data$label %in% excluded_and_not_present_species_labels, 'PR_AUC'])
  summary(data[data$model == "source" & !data$label %in% excluded_and_not_present_species_labels, 'f1_max'])
  summary(data[data$model == "source" & !data$label %in% excluded_and_not_present_species_labels, 'r_Tp'])
  summary(data[data$model == "source" & !data$label %in% excluded_and_not_present_species_labels, 'p_Tf1'])
  summary(data[data$model == "source" & !data$label %in% excluded_and_not_present_species_labels, 'r_Tf1'])
  
  write.csv(data, 'data/results/table_S3.csv', row.names = FALSE)
}
