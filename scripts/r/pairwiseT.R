setwd("/Users/Ryan/datamaster/data")
library(stargazer)
library(ggplot2)
library(gdata)
library(xtable)

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

dat = dat[,!(names(dat) == "node")]
summary(dat)

normalize = function(vec) {
  log = log1p(vec)
  return((log-min(log))/(max(log)-min(log)))
}
dat["knowledge_forward_cites_norm"] = normalize(dat["knowledge_forward_cites"])
dat["knowledge_h_index_norm"] = normalize(dat["knowledge_h_index"])
summary(dat)

## is it close to normal?
to_plot = data.frame(xx=c(dat$knowledge_h_index_norm, dat$knowledge_forward_cites_norm), yy=rep(c("H Index","Forward Cites"), each=length(log_h)))
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

ptt = pairwise.t.test(dat$knowledge_forward_cites, dat$source)
ptt

sink(sprintf("r/pairwise_t.tex"), append=FALSE, split=FALSE)
xtable(ptt$p.value, caption=ptt$method, digits=-3)
sink()
