#
# NOTE: Ensure you have run test_validate_and_evaluate_perf.py and test_evalute_label_confusion.py for necessary data

# https://r-graph-gallery.com/309-intro-to-hierarchical-edge-bundling.html

### VISUALIZE THE HIERARCHY
library(ggraph)
library(igraph)
library(dplyr)
library(cowplot)
library(scales)

node_alpha_max = 0.2

labels = read.csv('data/class_labels.csv')

confusion_mtx = read.csv('data/annotations/processed/confusion_matrix.csv', row.names=1, check.names = FALSE)

perf_metrics = read.csv('/Users/giojacuzzi/Downloads/performance_metrics.csv')

model_to_evaluate = 'data/test/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/pre-trained' # e.g. 'data/test/custom_S1_N125_LR0.001_BS10_HU0_LSFalse_US0_I0/custom'

perf_metrics = perf_metrics[perf_metrics$model == model_to_evaluate,]
perf_metrics$label = tolower(perf_metrics$label)
perf_metrics$PR_AUC[is.na(perf_metrics$PR_AUC)] <- 0.5 # override missing AUC values for absent classes for visibility

labels_to_remove = c("long-eared owl", "american three-toed woodpecker", "cassin's vireo", "western kingbird", "eastern kingbird", "dusky flycatcher", "mountain bluebird", "vesper sparrow", "american redstart", "cassin's finch", "clark's nutcracker", "pine grosbeak", "lazuli bunting", "bushtit", "ring-necked pheasant", "california quail")
confusion_mtx <- confusion_mtx[, !names(confusion_mtx) %in% labels_to_remove]
confusion_mtx <- confusion_mtx[!rownames(confusion_mtx) %in% labels_to_remove, ]

# labels_to_keep = c("") # c()
# if length(labels_to_keep) > 0 {
#   confusion_mtx <- confusion_mtx[rownames(confusion_mtx) %in% labels_to_remove, ]
# }
 
get_group = function(x) {
    if (grepl("biotic", x)) {
      strsplit(x, " ")[[1]][1]
    } else if (grepl("artifact", x)) {
      "artifact"
    } else if (grepl("other ", x)) {
      "abiotic"
    } else if (x == "origin") {
      NA
    } else {
      "biotic (source)"
    }
}

get_order = function(x) {
  if (grepl("other ", x)) {
    y = "Abiotic"
  } else {
    y = labels[labels['label'] == x, 'order'][1]
  }
  return(y)
}

get_family = function(x) {
  if (grepl("other ", x)) {
    y = "Other"
  } else{
    y = labels[labels['label'] == x, 'family'][1] 
  }
  return(y)
}

get_presence = function(x) {
  p = ifelse(perf_metrics[perf_metrics$label == x, 'N_pos'] > 0, 1.0, 0.0)
  if (length(p) == 0) {
    return(0)
  }
  return(p)
}

# groups = tools::toTitleCase(sapply(rownames(confusion_mtx), get_group)) # Abiotic, Biotic, Artifact, etc.
groups = sapply(rownames(confusion_mtx), get_order)
groups = ifelse(is.na(groups), "Other", groups)
# groups = sapply(rownames(confusion_mtx), get_family) 


my_d1 = data.frame(from="origin", to=unique(groups))
my_d2 = data.frame(from = unname(groups), to = names(groups))
# my_d2 = my_d2[order(my_d2$from), ]
my_d2$family = sapply(my_d2$to, get_family)
my_d2 = my_d2 %>% arrange(from, family, to) %>% select(-family)
priority_levels <- c("Artifact", "Abiotic", "Biotic")
my_d2$priority <- ifelse(my_d2$from %in% priority_levels, match(my_d2$from, priority_levels), length(priority_levels) + 1)
my_d2 <- my_d2[order(my_d2$priority), ]
my_d2 = my_d2 %>% select(-priority)

my_hierarchy <- rbind(my_d1, my_d2)

my_edges <- rbind(my_d1, my_d2)

# create a vertices data.frame. One line per object of our hierarchy, giving features of nodes.
my_vertices <- data.frame(name = unique(c(as.character(my_hierarchy$from), as.character(my_hierarchy$to))) )
my_vertices$Group = sapply(my_vertices$name, get_order)
my_vertices$Presence = as.numeric(sapply(my_vertices$name, get_presence))
my_vertices[my_vertices$Group %in% c('Abiotic', 'Biotic'), 'Presence'] = 1

# get PR AUC
library(dplyr)
my_vertices <- my_vertices %>%
  left_join(perf_metrics %>% select(label, PR_AUC), by = c("name" = "label")) %>%
  rename(AUC = PR_AUC) %>%
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
my_vertices$angle <- 90.0 - 360.0 * my_vertices$id / my_nleaves

# calculate the alignment of labels: right or left
# If I am on the left part of the plot, my labels have currently an angle < -90
my_vertices$hjust <- ifelse( my_vertices$angle < -90.0, 1, 0)

# flip angle BY to make them readable
my_vertices$angle <- ifelse(my_vertices$angle < -90.0, my_vertices$angle+180.0, my_vertices$angle)

# Create a graph object
my_mygraph <- igraph::graph_from_data_frame( my_edges, vertices=my_vertices )

# The connection object must refer to the ids of the leaves:
my_from  <-  match( my_connect$from, my_vertices$name)
my_to  <-  match( my_connect$to, my_vertices$name)

g = ggraph(my_mygraph, layout = 'dendrogram', circular = TRUE) + 
  geom_conn_bundle(data = get_con(from = my_from, to = my_to), alpha=0.35, width=0.5, aes(colour=..index..), tension = 0.98) +
  scale_edge_colour_distiller(palette = "RdPu") +
  # scale_edge_color_continuous(low="red", high="snow2") + # aliceblue, azure, snow2
  geom_node_text(aes(x = x*1.12, y=y*1.12, filter = leaf, label=tools::toTitleCase(name), angle = angle, hjust=hjust, colour=Group), size=2, alpha=1) +
  
  geom_node_point(aes(filter = leaf, x = x*1.07, y=y*1.07, colour=Group, size=AUC, alpha=Presence)) +
  scale_alpha(range=c(0.2,0.9)) + # for nodes
  # scale_colour_manual(values=c('dodgerblue','darkgray','purple','tomato'))+
  scale_color_manual(values=c("#777777", "#F8766D", "#DE8C00", "#222222", "#00BA38", "#00C08B", "#00BFC4", "#00B4F0", "#619CFF", "#C77CFF", "#F564E3", "#FF64B0")) +
  # scale_colour_manual(values=c('dodgerblue','tomato','darkorange','darkgray', 'purple', 'goldenrod', 'palegreen3', 'palegreen4', 'seagreen4', 'blue', 'royalblue1', 'black', 'pink'))+
  # scale_colour_manual(values= rep( brewer.pal(9,"Paired") , 30)) +
  scale_size_continuous( range = c(4.75,0.25) ) +
  
  theme_void() +
  expand_limits(x = c(-1.75, 1.75), y = c(-1.75, 1.75))

plot(g + theme(
  legend.position="none",
  # plot.margin=unit(c(margin,margin,margin,margin),"cm"),
))
legend = cowplot::get_legend(g)
plot_grid(legend, ncol = 1)

ggsave(file="data/figures/confusion.svg", plot=g + theme(legend.position="none"), width=8, height=8)
ggsave(file="data/figures/confusion_legend.svg", plot=legend, width=8, height=8)
