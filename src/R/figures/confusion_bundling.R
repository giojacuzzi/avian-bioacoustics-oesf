#
# NOTE: Ensure you have run test_validate_and_evaluate_perf.py and test_evalute_label_confusion.py for necessary data

# https://r-graph-gallery.com/309-intro-to-hierarchical-edge-bundling.html

### VISUALIZE THE HIERARCHY
library(ggraph)
library(igraph)

node_alpha_max = 0.2

confusion_mtx = read.csv('data/annotations/processed/confusion_matrix.csv', row.names=1, check.names = FALSE)

perf_metrics = read.csv('data/annotations/processed/label_perf_metrics.csv')

get_group = function(x) {
    if (grepl("_", x)) {
      strsplit(x, "_")[[1]][1]
    } else if (x == "origin") {
      NA
    } else {
      "biotic (source)"
    }
}

get_presence = function(x) {
  ifelse(perf_metrics[perf_metrics$label == x, 'N_P'] > 0, 1.0, 0.0)
}

groups = tools::toTitleCase(sapply(rownames(confusion_mtx), get_group))

my_d1 = data.frame(from="origin", to=unique(groups))
my_d2 = data.frame(from = unname(groups), to = names(groups))
my_d2 = my_d2[order(my_d2$from), ]
my_hierarchy <- rbind(my_d1, my_d2)

my_edges <- rbind(my_d1, my_d2)

# create a vertices data.frame. One line per object of our hierarchy, giving features of nodes.
my_vertices <- data.frame(name = unique(c(as.character(my_hierarchy$from), as.character(my_hierarchy$to))) )
my_vertices$Group = sapply(my_vertices$name, get_group)
my_vertices$Presence = as.numeric(sapply(my_vertices$name, get_presence))

# get PR AUC
library(dplyr)
my_vertices <- my_vertices %>%
  left_join(perf_metrics %>% select(label, AUC.PR), by = c("name" = "label")) %>%
  rename(AUC = AUC.PR) %>%
  mutate(
    Presence = ifelse(is.na(Presence), 1.0, Presence),
    AUC = ifelse(is.na(AUC), 1.0, AUC)
  )

# Create a graph object with the igraph library
my_mygraph <- graph_from_data_frame( my_hierarchy, vertices=my_vertices )
# This is a network object, you visualize it as a network like shown in the network section!

# With igraph: 
plot(my_mygraph, vertex.label="", edge.arrow.size=0, vertex.size=2)

# With ggraph:
ggraph(my_mygraph, layout = 'dendrogram', circular = FALSE) + 
  geom_edge_link() +
  theme_void()

ggraph(my_mygraph, layout = 'dendrogram', circular = TRUE) + 
  geom_edge_diagonal() +
  theme_void()

from <- rownames(confusion_mtx)
to <- colnames(confusion_mtx)

# # Create a dataframe with all combinations of row and column names
# result <- expand.grid(from = rownames(confusion_mtx), to = colnames(confusion_mtx))
# 
# # Add the values from the connection matrix
# result$value <- as.vector(as.matrix(confusion_mtx))

library(reshape2)

my_connect <- melt(as.matrix(confusion_mtx), varnames = c("from", "to"), value.name = "value")
my_connect = my_connect[my_connect$value > 0, ]
my_connect = my_connect[my_connect$from != my_connect$to, ]

library(ggraph)
library(igraph)
library(tidyverse)
library(RColorBrewer)

#Let's add information concerning the label we are going to add: angle, horizontal adjustement and potential flip
#calculate the ANGLE of the labels
my_vertices$id <- NA
my_myleaves <- which(is.na( match(my_vertices$name, my_edges$from) ))
my_nleaves <- length(my_myleaves)
my_vertices$id[ my_myleaves ] <- seq(1:my_nleaves)
my_vertices$angle <- 90 - 360 * my_vertices$id / my_nleaves

# calculate the alignment of labels: right or left
# If I am on the left part of the plot, my labels have currently an angle < -90
my_vertices$hjust <- ifelse( my_vertices$angle < -90, 1, 0)

# flip angle BY to make them readable
my_vertices$angle <- ifelse(my_vertices$angle < -90, my_vertices$angle+180, my_vertices$angle)

# Create a graph object
my_mygraph <- igraph::graph_from_data_frame( my_edges, vertices=my_vertices )

# The connection object must refer to the ids of the leaves:
my_from  <-  match( my_connect$from, my_vertices$name)
my_to  <-  match( my_connect$to, my_vertices$name)

ggraph(my_mygraph, layout = 'dendrogram', circular = TRUE) + 
  geom_conn_bundle(data = get_con(from = my_from, to = my_to), alpha=0.35, width=0.4, aes(colour=..index..), tension = 0.9) +
  # scale_edge_colour_distiller(palette = "RdPu") +
  scale_edge_color_continuous(low="red", high="aliceblue") + # aliceblue, azure
  geom_node_text(aes(x = x*1.15, y=y*1.15, filter = leaf, label=name, angle = angle, hjust=hjust, colour=Group), size=2, alpha=1) +
  
  geom_node_point(aes(filter = leaf, x = x*1.07, y=y*1.07, colour=Group, size=AUC, alpha=Presence)) +
  scale_alpha(range=c(0.2,0.7)) +
  scale_colour_manual(values=c('dodgerblue','darkgray','purple','tomato'))+
  # scale_colour_manual(values= rep( brewer.pal(9,"Paired") , 30)) +
  scale_size_continuous( range = c(6,0.5) ) +
  
  theme_void() +
  theme(
    legend.position="none",
    plot.margin=unit(c(0.2,0.2,0.2,0.2),"cm"),
  ) +
  expand_limits(x = c(-1.3, 1.3), y = c(-1.3, 1.3))

