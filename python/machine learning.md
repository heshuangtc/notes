### basic info
* store a model object
  - save a pickle file [pickle doc](https://docs.python.org/2/library/pickle.html)
    ```
    import pickle
    output = open('./path/filename.pkl', 'wb')
    pickle.dump(model, output)
    output.close()
    ```
  - load a pickle file
    ```
    import pickle
    pkl_file = open('./path/filename.pkl', 'rb')
    model = pickle.load(pkl_file)
    pkl_file.close()
    ```
*

### data preparation
* split data frame into train/test
  - sklearn [link](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
  ```
  from sklearn.model_selection import train_test_split
  X_train, X_test= train_test_split(X, test_size=0.33, random_state=42)
  ```
  - 
*

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
* sklearn clustering list [link](http://scikit-learn.org/stable/modules/clustering.html)
* kmeans
  - sklearn [link](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  ```
  from sklearn.cluster import KMeans
  KMeans(n_clusters=2, random_state=0).fit(df)
  kmeans.labels_
  ```

*

### Regression
* linear regression [link](http://scikit-learn.org/stable/modules/linear_model.html)
  - linear regression
  ```
  from sklearn import linear_model
  reg = linear_model.LinearRegression()
  reg.fit(df_train)
  reg.predict(df_test)
  ```
  - Ridge regression: Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of coefficients. 
  `reg = linear_model.Ridge (alpha = .5)`
  - Lasso regression:The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer parameter values, effectively reducing the number of variables upon which the given solution is dependent. 
  `reg = linear_model.Lasso(alpha = 0.1)`
  - logistic regression
  ```
  reg = linear_model.LogisticRegression()
  ```
* Gradient Boosting Regression [link](http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html)
  - ensemble.GradientBoostingRegressor
  ```
  from sklearn import ensemble
  params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
  clf = ensemble.GradientBoostingRegressor(**params)
  clf.fit(X_train, y_train)
  clf.predict(X_test)
  clf.feature_importances_
  ```
* grid search cv
  - linear regression
  ```
  from sklearn.model_selection import GridSearchCV
  from sklearn import linear_model
  reg0 = linear_model.LinearRegression()
  parameters = {'fit_intercept':[True,False], 'normalize':[True,False]} 
  #all parameters for linear regression
  reg = GridSearchCV(reg0,parameters)
  reg.fit(df_train)
  df_test['predict'] = reg.predict(df_test)
  ```
  - gbr
  ```
  parameters_gbr = {'n_estimators': [5,10], 'max_depth': [3,4], 'min_samples_split': [2,5]}
  model_gbr0 = ensemble.GradientBoostingRegressor()
  model_gbr = GridSearchCV(model_gbr0, parameters_gbr)
  model_gbr.fit(df_train[ls_train_cols], df_train.target_col)
  df_test['predict'] = model_gbr.predict(df_test[ls_train_cols])
  ```
*


### classification



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


### validation
* mean squared error
  - sklearn
    ```
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_real, y_predict)
    print("MSE: %.4f" % mse)
    ```
  - 
* r2/variance
  - linear `model_linear.score(df_test[ls_train_cols],df_test.hist_target)`

* cross validation
  - what is over-fitting [link](https://elitedatascience.com/overfitting-in-machine-learning#how-to-detect)
  - on train side, by given model object, dependent variables, target variable, predict and calculate accuracy rate
  ```
  from sklearn.model_selection import cross_val_score
  cross_val_score(model_object, X_train[['dayofweek','month','year','dayofmonth']], X_train[['order_count']], cv=5)
  ```
  - on test side, by given model object, dependent variables, target variable, predict and calculate accuracy rate
  ```
  from sklearn.model_selection import cross_val_score
  cross_val_score(model_object, X_test[['dayofweek','month','year','dayofmonth']], X_test[['order_count']], cv=5)
  ```