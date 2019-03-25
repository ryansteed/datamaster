setwd("/Users/Ryan/datamaster/data")
library(gdata)

# Input
engines = read.csv(sprintf("colonial/FEATURE_both_None_engines_05Mar19.csv"), sep=",", header=TRUE)
radio = read.csv(sprintf("colonial/FEATURE_both_None_radio_05Mar19.csv"), sep=",", header=TRUE)
robots = read.csv(sprintf("colonial/FEATURE_both_None_robots_05Mar19.csv"), sep=",", header=TRUE)
transportation = read.csv(sprintf("colonial/FEATURE_both_None_transportation_05Mar19.csv"), sep=",", header=TRUE)
xray = read.csv(sprintf("colonial/FEATURE_both_None_xray_05Mar19.csv"), sep=",", header=TRUE)
coherentlight = read.csv(sprintf("colonial/FEATURE_both_None_coherent-light_05Mar19.csv"), sep=",", header=TRUE)

names(engines)
names(radio)
names(robots)
names(transportation)
names(xray)
names(coherentlight)


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

dat = dat[,!(names(dat) == "node")]

# process dates
dat[,"patent_date"] = as.Date(dat[,"patent_date"])
## center dates and convert to diff time in weeks since min
dat[,"patent_date_center"] = as.numeric(dat[,"patent_date"] - min(dat[,"patent_date"]), units="weeks")

levels_to_list = function(vec, convertToInteger) {
  strList = lapply(
    vec, 
    function(x) 
      lapply(
        strsplit(gsub("\\]", "", (gsub("\\[", "", as.character(x)))), " "), 
        function(y) 
          unlist(lapply(y[!y==""], function(z) if (convertToInteger) strtoi(z) else z))
      )
  )
  return(strList)
}

for (list_dat in c("inventor", "assignee")) {
  total_num = paste(list_dat,"total_num_patents",sep="_")
  dat[total_num] = levels_to_list(dat[total_num], TRUE)
  dat[paste(list_dat,"max_num_patents", sep="_")] = sapply(dat[,total_num], function(x) max(unlist(x))) 
}

summary(dat)

