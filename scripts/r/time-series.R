setwd("/Users/Ryan/datamaster/data")
dat = read.table("colonial/FEATURES_both_None_coherent-light_25Feb19.csv", sep=",", header=TRUE)
head(dat)
# process dates
dat[,"patent_date"] = as.Date(dat[,"patent_date"])
## center dates and convert to diff time in weeks since min
dat[,"patent_date_center"] = as.numeric(dat[,"patent_date"] - min(dat[,"patent_date"]), units="weeks")

plot(dat[,'t'] + dat[,'t']^2 + dat[,'t']^3, log(dat[,'knowledge_forward_cites']))
fit = lm(
  log(k) ~ t + t_0 + patent_num_citedby_us + patent_processing_time + patent_num_claims, 
  data=dat
)
summary(fit)

# dat_final = c(nrow(dat))
# k=0
# j=0
# for (i in 1:nrow(dat)) {
#   if (dat[i,"t"]==0) {
#     l=i-1
#     # log previous data point as index of a final result
#     dat_final = append(dat_final,l)
#     # run a regression on 
#     if (i>1) {
#       fit = lm(log(k) ~ t, data=dat[c(k:l),])
#       coeff = coefficients(fit)
#       if (coeff['t']>.36) {
#         print("Found max")
#         plot(dat[c(k:l),'t'], dat[c(k:l),'k'])
#       }
#       if(j==0) {
#         coeffs = coeff
#       }
#       else {
#         coeffs = rbind(coeffs, coeff) 
#       }
#       k=i
#       j=j+1 
#     }
#   }
# }

# head(coeffs)
# summary(coeffs[,'t'])
# hist(coeffs[,'t'])
# max(coeffs[,'t'])


# subsample for aia implementation
dat_before = dat[dat$patent_date > "2011-09-16",]
dat_after = dat2[dat$patent_date < "2011-09-16",]
summary(dat_before[,"patent_date_center"])
fit = lm(
  log(k) ~ patent_date + patent_num_citedby_us,
  data=dat2_before
)
summary(fit)
summary(dat2_after[,"t_0_center"])
fit = lm(
  log(k) ~ t + t_0_center + patent_num_citedby_us,
  data=dat2_after
)
summary(fit)