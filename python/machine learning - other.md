## ------- basic info -------
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
* grid search cv
  - this is to try parameters combinations in each model to find the better model
  - linear regression
    ```
    from sklearn.model_selection import GridSearchCV
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    parameters = {'fit_intercept':[True,False], 'normalize':[True,False]} 
    #all parameters for linear regression
    reg = GridSearchCV(model,parameters,cv=5,n_jobs=-1)
    reg.fit(df_train)
    df_test['predict'] = reg.predict(df_test)
    ```
  - gbr
    ```
    parameters_gbr = {'n_estimators': [5,10], 'max_depth': [3,4], 'min_samples_split': [2,5]}
    model = ensemble.GradientBoostingRegressor()
    model_gbr = GridSearchCV(model, parameters_gbr)
    model_gbr.fit(df_train[ls_train_cols], df_train.target_col)
    df_test['predict'] = model_gbr.predict(df_test[ls_train_cols])
    ```
* lightgbm
  ```
  import lightgbm as lgb
  param = {'num_leaves':31, 'num_trees':100, 'objective':'binary',
    'metric':['auc', 'binary_logloss']}
  model = lgb.cv(param,dftrain,num_round=10,nfold=5)
  ```




## ------- data preparation -------
* split data frame into train/test
  - sklearn [link](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
  ```
  from sklearn.model_selection import train_test_split
  X_train, X_test= train_test_split(X, test_size=0.33, random_state=42)
  ```
  - 






## ------- normalization / feature scaling -------
* feature scaling will affect svm(rbf kernel) and kmeans instead of decision tree or linear regression
* normalization with sklearn
  - L2
    ```
    from sklearn.preprocessing import normalize
    df = normalize(df,method='l2',copy=True)
    ```
  - min/max sklearn
    ```
    from sklearn.preprocessing import MinMaxScaler
    MinMaxScaler.fit_transform(df.col)
    ```
* normalization with equivalent
  * z score `(df-df.mean())/df.std()`
  * min max `(df-df.min())/(df.max()-df.min())`






## ------- validation -------
* mean squared error
  - lower better
  - sklearn
    ```
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_real, y_predict)
    print("MSE: %.4f" % mse)
    ```
  - 
* r2/variance
  - linear `model_linear.score(df_test[ls_train_cols],df_test.hist_target)`
* score
  - precision, recall,fscore,support
    ```
    from sklearn.metrics import precision_recall_fscore_support as score
    precision,recall,fscore,support = score(ytest,ypredict,pos_label='yes',average='binary')
    ```
      + pos_label: the label to measure
      + average:
  - accuracy score
    ```
    from sklearn.metrics import accuracy_score
    accuracy_score(ypredict,ylabel)
    ```
* cross validation
  - what is over-fitting [link](https://elitedatascience.com/overfitting-in-machine-learning#how-to-detect)
  - kfold
    ```
    from sklearn.model_selection import KFold,cross_val_score
    cross_val_score(model_object,dffeatures,dflabel,cv=KFold(n_splits=5))
    # cv: select which index/row
    ```
  - on train side, by given model object, dependent variables, target variable, predict and calculate accuracy rate
    ```
    from sklearn.model_selection import cross_val_score
    cross_val_score(model_object, X_train[['dayofweek','month','year','dayofmonth']], X_train[['order_count']], cv=5, n_jobs=-1, scoring='accuracy')
    ```
    + scoring: accuracy, precision, recall
  - on test side, by given model object, dependent variables, target variable, predict and calculate accuracy rate
    ```
    from sklearn.model_selection import cross_val_score
    cross_val_score(model_object, X_test[['dayofweek','month','year','dayofmonth']], X_test[['order_count']], cv=5)
    ```
  - classification report (precision,recall,f1-score,support)
  ```
  from sklearn.metrics import classification_report
  classification_report(y,ypredict)
  ```



## optimization
* find minimum goal by tweaking numbers `import scipy; fmin_cg()`