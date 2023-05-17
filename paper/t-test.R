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
mn1 = c(75.10 , 1.44 , 43.33 , 11.11 , 35.41 , 38.48 , 32.78 , 18.13 , 32.43 , 36.54 , 1.30  , 6.59)
sd1 = c(2.59  , 2.22 , 1.52  , 3.52  , 5.02  , 1.95  , 3.59  , 1.55  , 5.84  , 3.53  , 00.87 , 2.79)
#table 2
mn2 = c(76.52 , 2.34 , 42.57 , 11.42 , 36.02 , 36.61 , 31.96 , 17.97 , 25.59 , 30.25 , 6.40 , 14.62)
sd2 = c(4.92  , 1.71 , 3.37  , 2.60  , 6.61  , 1.44  , 1.28  , 0.61  , 5.15  , 3.03  , 0.70 , 2.48)
#table 3
mn3 = c(78.42 , 2.78 , 45.76 , 11.34 , 36.79 , 36.89 , 32.04 , 18.01 , 28.87 , 30.80 , 6.60 , 15.29)
sd3 = c(5.77  , 1.88 , 2.94  , 4.98  , 7.61  , 1.68  , 1.16  , 0.51  , 4.86  , 4.44  , 0.66 , 4.78)
#table 4
mn4 = c(77.92 , 2.34 , 42.36 , 11.26 , 36.44 , 36.48 , 32.39 , 18.08 , 25.14 , 30.35 , 6.54 , 14.66)
sd4 = c(3.42  , 2.08 , 4.14  , 3.72  , 4.60  , 2.08  , 1.09  , 0.96  , 6.42  , 3.86  , 0.49 , 3.96)
#table 5
mn5 = c(75.54 , 2.11 , 42.57 , 11.03 , 33.40 , 36.37 , 30.46 , 17.67 , 24.69 , 26.05 , 6.26 , 13.24)
sd5 = c(4.91  , 1.65 , 3.45  , 4.59  , 6.13  , 2.49  , 1.03  , 0.79  , 5.29  , 2.77  , 1.01 , 2.62)
#table 6 (Baeline+PRP)
mn6.1 = c(76.67 , 2.49 , 42.84 , 11.60 , 36.37 , 36.73 , 31.58 , 18.01 , 24.86 , 32.06 , Inf , 14.29)
sd6.1 = c(4.24  , 1.7  , 4.64  , 4.83  , 7.03  , 2.18  , 1.45  , 0.86  , 6.02  , 2.93  , Inf , 3.55)
#table 6 (Baeline+PRP+LR)
mn6.2 = c(76.17 , 2.33 , 44.13 , 12.08 , 35.63 , 36.63 , 45.19 , 17.98 , 24.41 , 44.31 , Inf , 14.29)
sd6.2 = c(0.92  , 2.49 , 3.68  , 3.80  , 6.27  , 2.07  , 0.96  , 1.47  , 4.42  , 5.53  , Inf , 4.83)



getsf <- function(mx,sx,my,sy) {
  for (i in 1:length(mx)) {
    m1 = mx[i]
    s1 = sx[i]
    m2 = my[i]
    s2 = sy[i]
    print(paste(i,t.test2(m1,m2,s1,s2)[4]))
  }
  flush.console()
}


getsf(mn1,sd1,mn2,sd2)
getsf(mn1,sd1,mn3,sd3)
getsf(mn1,sd1,mn4,sd4)
getsf(mn1,sd1,mn5,sd5)
getsf(mn1,sd1,mn6.1,sd6.1)
getsf(mn1,sd1,mn6.2,sd6.2)
