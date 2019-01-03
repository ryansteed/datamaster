setwd("/Users/Ryan/datamaster/data")
dat = read.table("timedata.tsv", sep="\t", skip=1)
colnames(dat) = c("t", "k")
head(dat)
# plot(dat['t'] + dat['t']^2, log(dat['k']))
fit = lm(log(k) ~ t + t^2, data=dat)
summary(fit)
# Other useful functions 
# coefficients(fit) # model coefficients
# confint(fit, level=0.95) # CIs for model parameters 
# fitted(fit) # predicted values
# residuals(fit) # residuals
# anova(fit) # anova table 
# vcov(fit) # covariance matrix for model parameters 
# influence(fit) # regression diagnostics

k=0
j=0
for (i in 1:nrow(dat)) {
  if (dat[i,1]==0) {
    if (i>1) {
      fit = lm(log(k) ~ t, data=dat[c(k:i),])
      coeff = coefficients(fit)
      # print(coeff)
      if(i==2) {
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
