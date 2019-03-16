setwd("/Users/Ryan/datamaster/data")
library(gdata)
library(stargazer)

# Input
engines = read.csv(sprintf("colonial/FEATURE_both_None_engines_12Mar19.csv"), sep=",", header=TRUE)
radio = read.csv(sprintf("colonial/FEATURE_both_None_radio_05Mar19.csv"), sep=",", header=TRUE)
robots = read.csv(sprintf("colonial/FEATURE_both_None_robots_05Mar19.csv"), sep=",", header=TRUE)
transportation = read.csv(sprintf("colonial/FEATURE_both_None_transportation_05Mar19.csv"), sep=",", header=TRUE)
xray = read.csv(sprintf("colonial/FEATURE_both_None_xray_05Mar19.csv"), sep=",", header=TRUE)
coherentlight = read.csv(sprintf("colonial/FEATURE_both_None_coherent-light_05Mar19.csv"), sep=",", header=TRUE)

names=c("Engines", "Radio", "Robots", "Transportation", "X-Ray", "Coherent Light")
dat = combine(
  engines,
  radio,
  robots,
  transportation,
  xray,
  #coherentlight,
  names=names,
  recursive=FALSE
)
dat = na.omit(dat[,!(names(dat) == "node")])
summary(dat)

sink(sprintf("r/features_all.tex"), append=FALSE, split=FALSE)
stargazer(dat[,!(names(dat) == "source")], summary.stat=c("mean", "sd", "min", "max"), title="")
sink()

# process dates
dat[,"patent_date"] = as.Date(dat[,"patent_date"])
## center dates and convert to diff time in weeks since min
dat[,"patent_date_center"] = as.numeric(dat[,"patent_date"] - min(dat[,"patent_date"], na.rm = TRUE), units="weeks")

levels_to_list = function(vec, convertToInteger) {
  strList = lapply(
    vec, 
    function(x) 
      lapply(
        strsplit(gsub("'", "", gsub("\\]", "", (gsub("\\[", "", as.character(x)))), " "), " "), 
        function(y) 
          unlist(lapply(y[!y==""], function(z) if (convertToInteger) strtoi(z) else z))
      )
  )
  return(strList)
}
levels_to_list(engines['inventor_total_num_patents'], TRUE)

for (list_dat in c("inventor", "assignee")) {
  total_num = paste(list_dat,"total_num_patents",sep="_")
  dat[total_num] = levels_to_list(dat[total_num], TRUE)
  dat[paste(list_dat,"max_num_patents", sep="_")] = sapply(dat[,total_num], function(x) max(unlist(x))) 
}
summary(dat)

fit = lm(
  knowledge_forward_cites ~ patent_num_claims + patent_processing_time + inventor_max_num_patents + assignee_max_num_patents,
  data=dat
)
summary(fit)

# subsample for aia implementation
dat_before = dat[dat$patent_date > "2011-09-16",]
dat_after = dat2[dat$patent_date < "2011-09-16",]
summary(dat_before[,"patent_date_center"])
fit = lm(
  log(k) ~ patent_date + patent_num_citedby_us,
  data=dat_before
)
summary(fit)
summary(fit)