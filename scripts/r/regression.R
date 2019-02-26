setwd("/Users/Ryan/datamaster/data")
dat = read.table("colonial/FEATURES_both_None_coherent-light_25Feb19.csv", sep=",", header=TRUE)
# process dates
dat[,"patent_date"] = as.Date(dat[,"patent_date"])
## center dates and convert to diff time in weeks since min
dat[,"patent_date_center"] = as.numeric(dat[,"patent_date"] - min(dat[,"patent_date"]), units="weeks")

# Other useful functions 
# coefficients(fit) # model coefficients
# confint(fit, level=0.95) # CIs for model parameters 
# fitted(fit) # predicted values
# residuals(fit) # residuals
# anova(fit) # anova table 
# vcov(fit) # covariance matrix for model parameters 
# influence(fit) # regression diagnostics

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
# viewing the patents without a time dimension - using only final knowledge impact score
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