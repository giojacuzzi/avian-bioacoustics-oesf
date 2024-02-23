data = read.csv('data/species/Species List - Potential species.csv')
data = data[,c('Common_Name', 'Season_Breeding')]

data = na.omit(data)
data = data[data$Season_Breeding != 'N/A', ]
data = data[data$Season_Breeding != '', ]

data = rbind(data, data.frame(
  Common_Name = c("_D2", "_D4", "_D6", "_D8"),
  Season_Breeding = c("22 Apr-2 May", "21 May-2 Jun", "18 Jun-29 Jun", "16 Jul-26 Jul")
))

data['Season_Start'] = as.Date(sapply(strsplit(data$Season_Breeding, "-"), `[`, 1), format = "%d %b")
data['Season_End'] = as.Date(sapply(strsplit(data$Season_Breeding, "-"), `[`, 2), format = "%d %b")

library(ggplot2)
# Create a new dataframe for plotting
plot_df <- data.frame(
  species = data["Common_Name"],
  start = data$Season_Start,
  end = data$Season_End,
  y = 1:nrow(data)
)

# Create the plot
ggplot(plot_df, aes(x = start, xend = end, y = y, yend = y, color = Common_Name)) +
  geom_segment(size = 5) +
  labs(title = "Date Range Visualization", x = "Date", y = "Row") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b %Y")

