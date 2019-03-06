setwd("/Users/Ryan/datamaster/data")
library(stargazer)
library(ggplot2)
library(gdata)
library(MASS)

# Input
engines = read.csv(sprintf("colonial/METRICS_both_None_engines_20Feb19.csv"), sep=",", header=TRUE)
radio = read.csv(sprintf("colonial/METRICS_both_None_radio_20Feb19.csv"), sep=",", header=TRUE)
robots = read.csv(sprintf("colonial/METRICS_both_None_robots_20Feb19.csv"), sep=",", header=TRUE)
transportation = read.csv(sprintf("colonial/METRICS_both_None_transportation_20Feb19.csv"), sep=",", header=TRUE)
xray = read.csv(sprintf("colonial/METRICS_both_None_xray_20Feb19.csv"), sep=",", header=TRUE)
coherentlight = read.csv(sprintf("colonial/METRICS_both_None_coherent-light_19Feb19.csv"), sep=",", header=TRUE)

names=c("Engines", "Radio", "Robots", "Transportation", "X-Ray", "Coherent Light")
dat = combine(
  engines,
  radio,
  robots,
  transportation,
  xray,
  coherentlight,
  names=names
)

print = dat[,!(names(dat) == "node")]
colnames(print) = c(
  "Forward Cites",
  "Backward Cites",
  "Family Size",
  "Number of Claims",
  "H Index",
  "Knowledge (H Index)",
  "Knowledge (Forward Cites)",
  "Dataset"
)
summary(print)

log_h = log1p(dat$knowledge_h_index)
log_f = log1p(dat$knowledge_forward_cites)
#sqrt_h = sqrt(dat$knowledge_h_index)
#sqrt_h = sqrt(dat$knowledge_forward_cites)

to_plot = data.frame(xx=c(log_h, log_f), yy=rep(c("H Index","Forward Cites"), each=length(log_h)))
to_plot['src'] = print['Dataset']
head(to_plot)

hp = ggplot(to_plot, aes("", xx, colour=src)) + 
  geom_boxplot() +
  ggtitle("Total Knowledge Contribution") +
  ylab("K") +
  guides(colour=FALSE) + 
  theme_bw()
gg = hp + facet_grid(yy ~ src)
gg
ggsave(sprintf("r/tkc_boxplots.png"), gg)
