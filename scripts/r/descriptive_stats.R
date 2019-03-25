setwd("/Users/Ryan/datamaster/data")
library(stargazer)
library(ggplot2)
library(cowplot)
library(gdata)

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

sink(sprintf("r/metrics_all.tex"), append=FALSE, split=FALSE)
stargazer(print[,!(names(dat) == "source")], summary.stat=c("mean", "sd", "min", "max"), title="")
sink()
