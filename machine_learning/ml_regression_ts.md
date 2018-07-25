## ------- regression algorithm -------

from blog [link](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)

* linear regression
  - estimate real values based on continuous variable(s). establish relationship between independent and dependent variables by fitting a best line. This best fit line is known as regression line and represented by a linear equation `Y= a*X + b`.
  - These coefficients a and b are derived based on minimizing the sum of squared difference of distance between data points and regression line.
  - `Y` Dependent Variable / `a` Slope / `X` Independent variable / `b` Intercept
  - Simple Linear Regression is characterized by one independent variable. And, Multiple Linear Regression is characterized by multiple independent variables.

* logistic regression [link](https://developers.google.com/machine-learning/crash-course/logistic-regression/calculating-a-probability)
  -  a classification not a regression algorithm. returns a probability and need classification threshold (also called the decision threshold).
  -  estimate discrete values ( Binary values like 0/1, yes/no, true/false ) based on given set of independent variable(s). In simple words, it predicts the probability of occurrence of an event by fitting data to a logit function (a sigmoid function to make sure output always falls between 0 and 1).
  -  the log odds of the outcome is modeled as a linear combination of the predictor variables.
  ```
    odds= p/ (1-p) = probability of event occurrence / probability of not event occurrence
    ln(odds) = ln(p/(1-p))
    logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk
  ```
  -  p is the probability of presence of the characteristic of interest. It chooses parameters that maximize the likelihood of observing the sample values rather than that minimize the sum of squared errors (like in ordinary regression).
* ordinary least squares (OLS) regression
  - min(sum(distance between data points and fit line))









## ------- time series algorithm -------

* basic term
  - trend: stable linear movement up or down
  - seasonality: cyclical fluctuations throughout a year (every few months)
  - cyclical: cross years (every few years)
  - error: part/value that not explained by trend or seasonality
  - stationary time series: mean and variance are stable
  - ETS (exponential smoothing)
  - ARIMA (auto regressive integrated moving average)
  - ACF(auto-correlation function)
    + how correlated a time series is with its past values
    + more over |0.3|, more correlated to itself. otherwise, it is a stationary time series
  - PACF(partial auto correlation function)
    + current value - correlation between 2 variables


* non-seasonal ARIMA
  - `ARIMA(p,d,q)`
  - p: AR : number of previous periods to use
  - d(difference): I : process which transform time series into a stationary time series
    + first difference = value - previous value
    + second difference = value of first difference - previous value of first difference
    + till get a stationary time series
  - q: MA : lag of error component
  - based on ACF
    + if lag-1 is positive, use AR(p)
    + if lag-1 is negative, use MA(q), q=k(drop sharply at lag-k)
    + rarely use both AR and MA
  - based on PACF
    + if correlation drop sharply at lag-k, use AR(p), p=k(drop sharply at lag-k)
    + if correlation drop gradual, use MA(q)
* seasonal ARIMA
  - `ARIMA(p,d,q)(P,D,Q)m`
  - m: number of periods in each season (for example, if monthly data and Jan, from Jan2015 to Jan2016, then m=12)
  - (P,D,Q): seasonal portion
![seas arima](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-regressionts_seasarima.png)

* ETS
  - weighted average of past observations
  - ETS = error?trend?seasonality
    + ?: combine those 3 with `+` or `*`
    + `+` linear: trend and seasonality are relative constant overtime
    + `*` exponential: trend and seasonality increase/decrease magnitude over time (for example, max/min distance of seasonality increase. trend slope increase)
![ets combine](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-regressionts_ets.png)
  - time series decomposition plot to display ETS
* ETS models
  - simple exponential smoothing method
    + used for no trend, no seasonality. 
    + use level only.
    + forecast = `a(1-a)^t * Yt`
    + a: alpha, smoothing parameter,between 0 and 1
    + t: number of periods
    + Yt: value of period t
  - holt's linear trend method (double exponential smoothing)
    + used for no seasonality.
    + use level and trend. exponential smoothing on both separately then add together
  - exponential trend method
    + used for no seasonality.
    + use level and trend. exponential smoothing on both separately then multiply together
    + similar to holt's linear trend method except multiply instead of add
  - damped trend method
  - holt-winters seasonal method
    + use level and trend. 
    + add trend.
    + add for stable seasonality OR multiply for changing seasonality.
  

* naive method: 
  - next data is the one before
  - seasonal naive: seasonal pattern is same

* validation
  - holdout period (6 months usually)
  - residual = observed value - forecast value
    + should be not correlated(check residual ACF plot)
    + should have (close)0 mean
  - error measurements (lower better)
    + mean error(ME): `mean(obs-fcst)`
    + mean percentage error(MPE): `mean((obs-fcst)/obs)`, this can compare different models as it is scale independent measurement
    + root mean squared error(RMSE): `mean(sd(obs-fcst))`
    + mean absolute error(MAE): `mean(|obs-fcst|)`
    + mean absolute percentage error(MAPE): `mean(|obs-fcst|/obs)`, this can compare different models as it is scale independent measurement
  - AIC(akaike information criterion)
    + can compare 2 different models
    + lower, better fit

>> reference: Udacity Time Series Forecasting