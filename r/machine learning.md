### statistic terms
* basic
  - mean `mean(a_list,na.rm=T)`
  - standard deviation `sd(a_list,na.rm=T)`
  - variance `var(sd(a_list,na.rm=T)`
  - interquartile/50% quartile/median `IQR(a_list,na.rm=T)`
  - quantile 20% 70% `quantile(a_list,probs=c(.2,.7))`
  - z score `scale(a_list)` or `x-mean(a_list)/sd(a_list)`
  - kurtosis `e1071::kurtosis`
  - covariance `cov(a_list,b_list)` relationship 2 variables (negative/positive)
  - correlation `cor(a_list,b_list)`
  - cumulative `ecdf(a_list)`

* 

* 

* 

* 

* 

#### variable detection

* `library(Boruta)`

  `Boruta(y~., data=na.omit(df))`

#### regression

* `lm(y~x1+x2+x3,data=df)` `summary(fit)`

* `glm(y~., data=df)`

#### classification

* Bayesian classification

  * `library(rpud)`

  `x1 <- rvbm.sample.train$X[, 1]`

  `x2 <- rvbm.sample.train$X[, 2]`

  `tc <- rvbm.sample.train$t.class`

  * library(vbmp)

  `res.vbmp <- vbmp(rvbm.sample.train$X, rvbm.sample.train$t.class, rvbm.sample.test$X, rvbm.sample.test$t.class, theta = rep(1., ncol(rvbm.sample.train$X)), control = list( sKernelType="gauss", bThetaEstimate=TRUE, bMonitor=TRUE, InfoLevel=1))`

  `covParams(res.vbmp) `

  `predError(res.vbmp)`

  `predLik(res.vbmp)`

* decision tree

  * `library(rpart)`

    `fit <- rpart(y~., data=df, method = 'anova')`

    `predict(fit,df2)`

  * `library(party)`

    `fit <- ctree(y~., data=df)`

    `predict(fit,df2)`




#### text mining

* `library(tm)`

  * build text mining object

    `review_source <- VectorSource(long_string_here)` or `review_source <- VectorSource(wordStem(long_string_here))`

    `corpus <- Corpus(review_source)`

  * clean string

    `toSpace <- content_transformer(function(x, pattern) {return (gsub(pattern, ' ', x))})`

    `corpus <- tm_map(corpus, toSpace, '/')`

    `corpus <- tm_map(corpus, toSpace, '-')`

    `corpus <- tm_map(corpus, toSpace, '[(]')`

    `corpus <- tm_map(corpus, toSpace, '["]')`

    `corpus <- tm_map(corpus, content_transformer(tolower))`

    `corpus <- tm_map(corpus, removePunctuation)`

    `corpus <- tm_map(corpus, stripWhitespace)`

  * clean string customized words

    `mystopwords <-c(stopwords("english"),1,2,3,4,5,6,'can','will','also')`

    `corpus <- tm_map(corpus, removeWords, mystopwords)`

  * create the document-term matrix

    `dtm <- DocumentTermMatrix(corpus)`

* `library(wordcloud)`

  `wordcloud(list_of_words, list_of_num, min.freq=2, colors=brewer.pal(6,'Dark2'))`

* `library(RWeka)`

  `BigramTokenizer <- NGramTokenizer(x, Weka_control(min = 2, max = 4)`

  `TermDocumentMatrix(corpus, control = list(tokenize = BigramTokenizer)`

#### optimization

* `library(lpSolve)`

  * sample 1

  `A1 <- cbind(diag(2*n),0) # One constraint per row: a[i], b[i] >= 0`

  `A2 <- cbind(diag(n), -diag(n), 1) # a[i] - b[i] = x[i] - mu`

  `r <- lp("min",c(rep(1,2*n),0),rbind(A1, A2),c(rep(">=", 2*n), rep("=", n)),c(rep(0,2*n), x))`

  `tail(r$solution,1)`

  `median(x)`

  * sample 2

  `tau <- .1`

  `r <- lp("min",c(rep(tau,n),rep(1-tau,n),0),rbind(A1, A2),c(rep(">=", 2*n), rep("=", n)),c(rep(0,2*n), x))`

  `tail(r$solution,1)`

  `quantile(x,tau)`

  * sample 3

  `tau <- .3`

  `n <- 100`

  `x1 <- rnorm(n)`

  `x2 <- rnorm(n)`

  `y <- 1 + x1 + x2 + rnorm(n)`

  `X <- cbind( rep(1,n), x1, x2 )`

  `A1 <- cbind(diag(2*n), 0,0,0)  # a[i], b[i] >= 0`

  `A2 <- cbind(diag(n), -diag(n), X) # a[i] - b[i] = (y - X %*% beta)[i]`

  `r <- lp("min",c(rep(tau,n), rep(1-tau,n),0,0,0),rbind(A1, A2),c(rep(">=", 2*n), rep("=", n)),c(rep(0,2*n), y))`

  `tail(r$solution,3)`

* basic pkg

  * sample 1

  `fr <- function(x) {x1 <- x[1];x2 <- x[2];100 * (x2 - x1 * x1)^2 + (1 - x1)^2}`

  `grr <- function(x) { ## Gradient of 'fr';x1 <- x[1];  x2 <- x[2];  c(-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1),200 *(x2 - x1 * x1))}`

  `optim(c(-1.2,1), fr)`

  `res <- optim(c(-1.2,1), fr, grr, method = "BFGS")`

  `optimHess(res$par, fr, grr)`

  `optim(c(-1.2,1), fr, NULL, method = "BFGS", hessian = TRUE)`
  
  
### time series

[link](https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/)

* moving average

  * TTR

    `SMA( array, n = avg.period)`

    `EMA( array, n = avg.period, wilder = T/F, ratio=NULL)`

  * ZOO

    `rollmean()`

    `rollmedian()`

    `rollsum(df$col, 2, fill=NA)`

* log/exp

  `log()`

  `exp()`

* `sd(df, na.rm = TRUE)`

* `scale(df,center = TRUE,scale=(sd(df,na.rm = TRUE)))`

* anova

  `tm1 = gl(k1, 1, n*k1*k2, factor(f1))`

  `tm2 = gl(k2, n*k1, n*k1*k2, factor(f2))`

  `aov(r ~ tm1 * tm2)`

  `summary(av)`