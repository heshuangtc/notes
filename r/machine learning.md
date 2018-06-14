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

* normal distribution
  - by given mean and sd to find observed value probability `pnorm(observed_value,mean,sd)`
  - by given mean and sd to find 95% value point/upper 5% `qnorm(.95,mean,sd)`
  - by given mean and sd with z score value to find p value `dnorm(z score,mean,sd)`

* exponential distribution
  - (also known as negative exponential distribution) is the probability distribution that describes the time between events in a Poisson point process
  - `dexp` gives the density(p value/significant value)
  - `pexp` gives the distribution function(percentage)
  - `qexp` gives the quantile function(score)
  - `rexp` generates random deviates()
  - by given mean and observed value to find probability `pexp(observed_value/mean)`

* z test
  - `BSDA::z.test(alist, sigma.x = sd)`

* ANOVA
  - basic info/example [link](http://personality-project.org/r/r.guide/r.anova.html#oneway)
  - 1way anova `aov(y~a+b,data=df)` y usually numeric data, a/b can be string

* chisquare
  - tbl is numeric table `library('MASS');chisq.test(tbl)`

* DW test
  - if there is a correlation in linear model
    ```
    linear_model <- lm(y~x,data=df)
    summary(linear_model)
    plot(linear_model_b)
    library(lmtest)
    dwtest(linear_model)
    ```

* log/exp

  `log()`

  `exp()`

* `sd(df, na.rm = TRUE)`

* `scale(df,center = TRUE,scale=(sd(df,na.rm = TRUE)))`


#### variable detection

* `library(Boruta)`

  `Boruta(y~., data=na.omit(df))`


#### regression

* linear regression
  - basic `lm(y~x1+x2+x3,data=df)` `summary(fit)`
  - confidence interval from lm `predict(lm,newdataframe,level=.95,interval='confidence')`
  - prediction interval from lm `predict(lm,data.frame(ROOMS=6),interval="prediction")`
  - plot linear regression
    `plot(df$y)`

    `abline(lm_model)` or `abline(lm_model$coefficients[1],lm_model$coefficients[2])`#slop is coefficients[2][2] if there is only one x variable

* `glm(y~., data=df)`

* non linear regression
  - nonlinear by given a and b start value and formula `nls(y~a*exp(-b*x),start=list(a=a_start,b=b_start))`


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

* random forest
  - randomforest
    ```
    library(randomForest)
    model = randomForest(y ~ . , data = df_train, ntree=10)
    model
    plot(model)
    pred = predict(model,df_test) 
    ```
  - 
* KNN
  - library(class) [link](https://stat.ethz.ch/R-manual/R-devel/library/class/html/knn.html) [sample](https://rstudio-pubs-static.s3.amazonaws.com/123438_3b9052ed40ec4cd2854b72d1aa154df9.html)
  - basic
    ```
    library(class)
    knn_model = knn(df_train,df_test,cl=as.factor(df_train$y),k=5)
    ```



### cluster
* kmeans
  - remove missing `na.omit(df)`
  - fit `fit = kmeans(df,4)` result`fit$cluster`
  - 


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

* lpSolve and lpSolveAPI package `library(lpSolve)`
  - pdf intro of lpSolveAPI [link](https://cran.r-project.org/web/packages/lpSolveAPI/lpSolveAPI.pdf)
  - pdf intro of lpSolve [link](https://cran.r-project.org/web/packages/lpSolve/lpSolve.pdf)
  - r blogger samples [link](http://horicky.blogspot.com/2013/01/optimization-in-r.html)

  - sample 1

  `A1 <- cbind(diag(2*n),0) # One constraint per row: a[i], b[i] >= 0`

  `A2 <- cbind(diag(n), -diag(n), 1) # a[i] - b[i] = x[i] - mu`

  `r <- lp("min",c(rep(1,2*n),0),rbind(A1, A2),c(rep(">=", 2*n), rep("=", n)),c(rep(0,2*n), x))`

  `tail(r$solution,1)`

  `median(x)`

  - sample 2

  `tau <- .1`

  `r <- lp("min",c(rep(tau,n),rep(1-tau,n),0),rbind(A1, A2),c(rep(">=", 2*n), rep("=", n)),c(rep(0,2*n), x))`

  `tail(r$solution,1)`

  `quantile(x,tau)`

  - sample 3

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

  - linear optimization (max)
    ```
    library(lpSolveAPI)
    op_model = make.lp(0,2) #2 decision variable
    set.objfn(op_model,c(-24,-40)) #max use negative coefficient
    add.constraint(op_model,c(18,12),'<=',4000)
    add.constraint(op_model,c(6,10),'<=',3500)
    add.constraint(op_model,c(1,0),'>=',0)
    add.constraint(op_model,c(0,1),'>=',0)
    solve(op_model)
    get.variables(op_model)
    get.objective(op_model)
    ```

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
* change data frame col to time series type/vector[link](https://www.statmethods.net/advstats/timeseries.html)
  - by month `myts <- ts(df$col, start=c(2009, 1), end=c(2014, 12), frequency=12)`
  - by day `myts <- ts(df$col, start=c(2009, 1), end=c(2009, 1), frequency=1)`
  - can only specify start date `myts <- ts(df$col, start=c(2009, 1), frequency=12)`

* Exponential Models[link](https://www.statmethods.net/advstats/timeseries.html)
  - simple exponential - models level
    `fit <- HoltWinters(myts, beta=FALSE, gamma=FALSE)`
  - double exponential - models level and trend
    `fit <- HoltWinters(myts, gamma=FALSE)`
  - triple exponential - models level, trend, and seasonal components
    `fit <- HoltWinters(myts)`


* moving average

  * TTR

    `SMA( array, n = avg.period)`

    `EMA( array, n = avg.period, wilder = T/F, ratio=NULL)`

  * ZOO

    `rollmean()`

    `rollmedian()`

    `rollsum(df$col, 2, fill=NA)`

* anova

  `tm1 = gl(k1, 1, n*k1*k2, factor(f1))`

  `tm2 = gl(k2, n*k1, n*k1*k2, factor(f2))`

  `aov(r ~ tm1 * tm2,data=df)`

  `summary(av)`

#### validation
* split data into train/test
    - by index
    ```
    list_sample <- sample(nrow(df),round(nrow(df)*0.7))
    df_train <- df[list_sample,]
    df_test <- df[-list_sample,]
    ```

* compare y and predict y
  - `cor(y,predict(model,newdata))` 