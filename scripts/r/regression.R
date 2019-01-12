setwd("/Users/Ryan/datamaster/data")
dat = read.table("colonial/FEATURES-TIME-DATA_12Jan19.tsv", sep="\t", skip=1)
head(dat)
colnames(dat) = c(
  "k", 
  "t", 
  "t_0", 
  "patent_num_claims", 
  "patent_num_citedby_us", 
  "patent_processing_time",
  "inventor_id", 
  "inventor_total_num_patents", 
  "inventor_key_id", 
  "assignee_type", 
  "assignee_total_num_patents", 
  "assignee_id", 
  "assignee_key_id", 
  "cpc_category", 
  "cpc_group_id", 
  "nber_category_id", 
  "nber_subcat_id", 
  "uspc_mainclass_id", 
  "uspc_subclass_id", 
  "ipc_class", 
  "ipc_main_group", 
  "wipo_field_id"
)
# process dates
dat[,"t_0"] = as.Date(dat[,"t_0"])
## center dates and convert to diff time in weeks since min
dat[,"t_0"] = as.numeric(dat[,"t_0"] - min(dat[,"t_0"]), units="weeks")
head(dat)

# plot(dat['t'] + dat['t']^2, log(dat['k']))
fit = lm(
  log(k) ~ t + t_0 + patent_num_citedby_us + patent_processing_time + patent_num_claims, 
  data=dat
)
summary(fit)
# Other useful functions 
# coefficients(fit) # model coefficients
# confint(fit, level=0.95) # CIs for model parameters 
# fitted(fit) # predicted values
# residuals(fit) # residuals
# anova(fit) # anova table 
# vcov(fit) # covariance matrix for model parameters 
# influence(fit) # regression diagnostics

dat_final = c(nrow(dat))
k=0
j=0
for (i in 1:nrow(dat)) {
  if (dat[i,"t"]==0) {
    l=i-1
    # log previous data point as index of a final result
    dat_final = append(dat_final,l)
    # run a regression on 
    if (i>1) {
      fit = lm(log(k) ~ t, data=dat[c(k:l),])
      coeff = coefficients(fit)
      if (coeff['t']>.36) {
        print("Found max")
        plot(dat[c(k:l),'t'], dat[c(k:l),'k'])
      }
      if(j==0) {
        coeffs = coeff
      }
      else {
        coeffs = rbind(coeffs, coeff) 
      }
      k=i
      j=j+1 
    }
  }
}
head(coeffs)
summary(coeffs[,'t'])
hist(coeffs[,'t'])
max(coeffs[,'t'])


# viewing the patents without a time dimension - using only final knowledge impact score
dat2 = dat[dat_final,]
summary(dat2[,"t_0"])
fit = lm(
  log(k) ~ t + t_0 + patent_num_citedby_us,
  data=dat2
)
summary(fit)