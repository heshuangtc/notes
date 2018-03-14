### normalization
* normalization with sklearn
  ```
  from sklearn.preprocessing import normalize
  df = normalize(df,method='l2',copy=True)
  ```
* normalization with equavilent
  * z score `(df-df.mean())/df.std()`
  * min max `(df-df.min())/(df.max()-df.min())`

###clustering
* pca with sklearn

*
###classification



### Time Series
#### moveing average
* pandas pkg
```
import pandas as pd
pd.rolling_mean(df.col1, avg_length)
```


#### holt-winters
* knowledge of method
actually it is triple exponential smoothing

* fecon235 pkg [link](https://github.com/rsvp/fecon235#dt_2015-08-01_094628)

* seasonal pkg [link](https://github.com/welch/seasonal/blob/master/examples/hw.py)

* pycast pkg
`pycast.methods.exponentialsmoothing.HoltWintersMethod`

#### auto regression
* knowledge of method

  * ARIMA: Auto-Regressive Integrated Moving Averages [link](https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/)

* [statsmodels ARIMA](http://www.statsmodels.org/0.6.1/generated/statsmodels.tsa.arima_model.ARIMA.html)

  * MA
  ```
  from statsmodels.tsa.arima_model import ARIMA
  model = ARIMA(ts_df, order=(0, 1, 2))  
  results_MA = model.fit(disp=-1) 
  ```
  
  * AR
  ```
  from statsmodels.tsa.arima_model import ARIMA
  model = ARIMA(ts_df, order=(2, 1, 0))  
  results_AR = model.fit(disp=-1)
  ```
  
* pyflux [link](http://www.pyflux.com/)
```
import pyflux as pf
model = pf.ARIMA(data=data, ar=4, ma=4, target='sunspot.year', family=pf.Normal())
x = model.fit("MLE")
x.summary()
model.predict(100) 
#or model.simulation_smoother(data, beta)
```

* Combined Model
```
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)
```