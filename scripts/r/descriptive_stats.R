setwd("/Users/Ryan/datamaster/data")
library(stargazer)
library(ggplot2)
library(cowplot)

# Input
date = "20Feb19"
query = "transportation"

dat = read.csv(sprintf("colonial/METRICS_both_None_%s_%s.csv", query, date), sep=",", header=TRUE)

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
summary(print)
sink(sprintf("r/metrics_%s.tex", query), append=FALSE, split=FALSE)
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
  theme(legend.position="top") +
  ylab("Density") +
  xlab("Log(K)")
hp
ggsave(sprintf("r/tkc_dist_%s.png", query), hp)

# to_plot2 = data.frame(xx=c(dat$knowledge_h_index, dat$knowledge_forward_cites), yy=rep(c("H Index","Forward Cites"),each=length(log_h)))
hp = ggplot(to_plot, aes(yy, xx)) + 
  geom_boxplot() +
  ggtitle("TKC Distribution By Weighting Method") +
  theme_classic() +
  theme(legend.title=element_blank()) +
  theme(text=element_text(size=12, family="Times New Roman")) +
  ylab("K") +
  xlab("Weighting Method")
hp
ggsave(sprintf("r/tkc_boxplot_%s.png", query), hp)
