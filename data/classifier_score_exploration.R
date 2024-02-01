folder_path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/annotation/data/raw_detections/2020/Deployment5/S4A04271_20200604'

csv_files <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)

# Initialize an empty list to store dataframes
dfs <- list()

# Loop through each CSV file and read them into the list
for (file in csv_files) {
  print(file)
  df <- read.csv(file)
  dfs <- append(dfs, list(df))
}

# Combine all dataframes into a single dataframe
data <- do.call(rbind, dfs)
data['start_date'] = as.POSIXct(data['start_date'], )

library(dplyr)
top_confidence_per_name <- data %>%
  group_by(common_name) %>%
  arrange(desc(confidence)) %>%
  slice_head(n = 1)
print(n = 116, top_confidence_per_name)

library(car)
species = "Anna's Hummingbird"
testy = data[data$common_name==species, ]
hist(testy$confidence)
hist(testy[testy$confidence>0.05, 'confidence'])
testy$logit = logit(testy$confidence)
hist(testy$logit)
hist(testy$logit, breaks=200)
hist(testy$logit, breaks=200, ylim=c(0,1000))
# logit_NZ = logit(testy[testy$confidence>0.05, 'confidence'])
# hist(logit_NZ)

