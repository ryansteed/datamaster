setwd("/Users/Ryan/datamaster/data")
library(stargazer)
library(ggplot2)
library(cowplot)
dat = read.csv("colonial/METRICS_both_None_coherent-light_17Feb19.csv", sep=",", header=TRUE)
print = dat[,!(names(dat) == "node")]
colnames(print) = c(
  "Forward Cites",
  "Backward Cites",
  "Family Size",
  "Number of Claims",
  "H Index",
  "Knowledge (H Index)",
  "Knowledge (Forward Cites)"
)
sink("r/metrics_coherent-light.tex", append=FALSE, split=FALSE)
stargazer(print, summary.stat=c("mean", "sd", "min", "max"), title="")
sink()

log_h = log(dat$knowledge_h_index)
log_f = log(dat$knowledge_forward_cites)
to_plot = data.frame(xx=c(log_h, log_f), yy=rep(c("H Index","Forward Cites"),each=length(log_h)))
head(to_plot)
# Density plots
hp = ggplot(to_plot, aes(x=xx, fill=yy, coulour=yy)) + 
  geom_density(alpha=0.7, position="stack") +
  ggtitle("TKC Distribution By Weighting Method") +
  theme_classic() +
  theme(legend.title=element_blank()) +
  theme(legend.position="top")
  ylab("Density") +
  xlab("Log(K)")
hp
ggsave("r/tkc_dist.png", hp)

# to_plot2 = data.frame(xx=c(dat$knowledge_h_index, dat$knowledge_forward_cites), yy=rep(c("H Index","Forward Cites"),each=length(log_h)))
hp = ggplot(to_plot, aes(yy, xx)) + 
  geom_boxplot() +
  ggtitle("TKC Distribution By Weighting Method") +
  theme_classic() +
  theme(legend.title=element_blank()) +
  ylab("K") +
  xlab("Weighting Method")
hp
ggsave("r/tkc_boxplot.png", hp)

# hp + facet_grid(. ~ yy)



#to_plot = data.frame(log_h, log_f)
#p1 = ggplot(to_plot, aes(x=log_f)) + stat_density(alpha=0.7) + theme_classic()
#p2 = ggplot(to_plot, aes(x=log_h)) + stat_density(alpha=0.7) + theme_classic()

#p = plot_grid(p1, p2)
#p
# save_plot("plot.pdf", p, ncol=2)
