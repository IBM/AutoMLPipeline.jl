# https://stats.stackexchange.com/questions/30394/how-to-perform-two-sample-t-tests-in-r-by-inputting-sample-statistics-rather-tha

# m1, m2: the sample means
# s1, s2: the sample standard deviations
# n1, n2: the same sizes
# m0: the null value for the difference in means to be tested for. Default is 0. 
# equal.variance: whether or not to assume equal variance. Default is FALSE. 
t.test2 <- function(m1,m2,s1,s2,n1=10,n2=10,m0=0,equal.variance=FALSE)
{
    if( equal.variance==FALSE ) 
    {
        se <- sqrt( (s1^2/n1) + (s2^2/n2) )
        # welch-satterthwaite df
        df <- ( (s1^2/n1 + s2^2/n2)^2 )/( (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) )
    } else
    {
        # pooled standard deviation, scaled by the sample sizes
        se <- sqrt( (1/n1 + 1/n2) * ((n1-1)*s1^2 + (n2-1)*s2^2)/(n1+n2-2) ) 
        df <- n1+n2-2
    }      
    t <- (m1-m2-m0)/se 
    dat <- c(m1-m2, se, t, 2*pt(-abs(t),df))    
    names(dat) <- c("Difference of means", "Std Error", "t", "p-value")
    return(dat) 
}

set.seed(0)
x1 <- rnorm(100)
x2 <- rnorm(200)
# you'll find this output agrees with that of t.test when you input x1,x2
(tt2 <- t.test2(mean(x1), mean(x2), sd(x1), sd(x2), length(x1), length(x2)))

(tt2 <- t.test2(26.05, 36.54,2.77 , 3.53, 5, 5))

#table 1
mn1 = c(75.10,1.44,43.33,11.11,35.41,38.48,32.78,18.13,32.43,36.54,1.30,6.59)
sd1 = c(2.59,2.22,1.52,3.52,5.02,1.95,3.59,1.55,5.84,3.53,00.87,2.79)
#table 5
mn2 = c(75.54,2.11,42.57,11.03,33.40,36.37,30.46,17.67,24.69,26.05,6.26,13.24)
sd2 = c(4.91,1.65,3.45,4.59,6.13,2.49,1.03,0.79,5.29,2.77,1.01,2.62)
#table 6
mn3 = c(76.67,2.49,42.84,11.60,36.37,36.73,31.58,18.01,24.86,32.06,Inf,14.29)
sd3 = c(4.24,1.7,4.64,4.83,7.03,2.18,1.45,0.86,6.02,2.93,Inf,3.55)
mn4 = c(76.17,2.33,44.13,12.08,35.63,36.63,45.19,17.98,24.41,44.31,Inf,14.29)
sd4 = c(0.92,2.49,3.68,3.80,6.27,2.07,0.96,1.47,4.42,5.53,Inf,4.83)



getsf <- function(mx1,sx1,my2,sy2) {
  for (i in 1:length(mx1)) {
    m1 = mx1[i]
    s1 = sx1[i]
    m2 = my2[i]
    s2 = sy2[i]
    print(paste(i,t.test2(m1,m2,s1,s2)[4]))
  }
}

getsf(mn1,sd1,mn2,sd2)
getsf(mn1,sd1,mn3,sd3)
getsf(mn1,sd1,mn4,sd4)
