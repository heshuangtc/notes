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
*


### classification
* random forest
  - sklearn
    ```
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_jobs=-1)
    ```
    + n_jobs: 1 run 1 tree each time, -1 run all trees parallel
    + max_depth: None till min leaf, 4 num of leaves
    + n_estimators: 10 trees in forest


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
* score
  - precision, recall,fscore,support
    ```
    from sklearn.metrics import precision_recall_fscore_support as score
    precision,recall,fscore,support = score(ytest,ypredict,pos_label='yes',average='binary')
    ```
      + pos_label: the label to measure
      + average:
  - 
* cross validation
  - what is over-fitting [link](https://elitedatascience.com/overfitting-in-machine-learning#how-to-detect)
  - kfold
    ```
    from sklearn.model_selection import KFold
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


## Natural Language Processing
* intro with python [link](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
* install packages
  - `pip install nltk` The complete toolkit for all NLP techniques.
  - `pip install pattern` A web mining module for the with tools for NLP and machine learning.
  - `pip install TextBlob` Easy to use nl p tools API, built on top of NLTK and Pattern.
  - `pip install Gensim` Topic Modelling for Humans
### feature engineering
* remove words
  - `import re;re.sub(string,'',source_str)`
  - `str.replace(string,'',source_str)`
* annoying words
  - special characters `import string;string.punctuation`
  - english stopwords `import nltk;nltk.corpus.stopwords.words('english')`
* lexicon normalization
  - Lemmatization
    ```
    from nltk.stem.wordnet import WordNetLemmatizer
    WordNetLemmatizer().lemmatize('multiplying', 'v')
    ## 'multiply'
    ```
  - Stemming
    ```
    from nltk.stem.porter import PorterStemmer
    PorterStemmer().stem('multiplying')
    ## 'multipli'
    ```
* speech tagging
  ```
  from nltk import word_tokenize, pos_tag
  pos_tag(word_tokenize(a_sentence))
  ```
* entity extraction
  - Latent Dirichlet Allocation (LDA)
    ```
    # convert str to matrix
    import corpora
    dictionary = corpora.Dictionary(str.split())
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in str.split()]

    # train matrix to get topics
    import gensim from gensim
    gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50).print_topics()
    ```
* statistic features/vectorization
  - count vectorization
    ```
    from sklearn.feature_extraction.text import CountVectorizer
    out = CountVectorizer().fit_transform(df[textcol]) #convert textcol to many columns and each col is one word
    out = CountVectorizer(analyzer=afun).fit_transform(df[textcol]) # apply customized functions before fit
    out.get_feature_names() #give all unique words
    # output will be sparse matrix so need to convert to df
    df = pd.DataFrame(out.toarray())
    ```
  - n-grams
    ```
    from sklearn.feature_extraction.text import CountVectorizer
    out = CountVectorizer(ngrams_range=(1,1)).fit_transform(df[textcol]) #default 1 word 1 column
    out = CountVectorizer(ngrams_range=(1,2)).fit_transform(df[textcol]) #1gram + bigrams
    out = CountVectorizer(ngrams_range=(2,2)).fit_transform(df[textcol]) #bigrams
    # output will be sparse matrix so need to convert to df
    df = pd.DataFrame(out.toarray())
    ```
  - Term Frequency – Inverse Document Frequency (TF – IDF)
    ```
    from sklearn.feature_extraction.text import TfidfVectorizer
    out = TfidfVectorizer().fit_transform(['a sentence', 'b sentence'])
    out = TfidfVectorizer(analyzer=customized_fun).fit_transform(df[textcol]) # customized_fun can do some cleaning or transformation before vectorized
    # output will be sparse matrix so need to convert to df
    df = pd.DataFrame(out.toarray())
    ```
* Word embedding
  - word2vec
    ```
    from gensim.models import Word2Vec
    model = Word2Vec([[w1,w2],[w1,w2,w3],[w3,w4]], min_count=1)
    model.similarity(w2, w3)
    ```
  - glove
    ```
    ```
* count text len without space
  - `df[textcol].apply(lambda x:len(x)-x.count(' '))`
  - `df[textcol].str.replace(' ','').apply(lambda x:len(x))`
### use case
* text classification
  - textblob (NaiveBayesClassifier)
    ```
    from textblob.classifiers import NaiveBayesClassifier as NBC
    from textblob import TextBlob
    # train
    model = NBC(dftrain) #df col1:str col2:label
    # predict
    model.classify('new str')
    # accuracy
    model.accuracy(dftest)
    ```
  - sklearn (svm)
    ```
    import TfidfVectorizer from sklearn.metrics
    train_vectors = TfidfVectorizer(min_df=4, max_df=0.9).fit_transform(dftrain) #dftrain without label
    test_vectors = TfidfVectorizer(min_df=4, max_df=0.9).fit_transform(dftest)

    from sklearn import svm
    model = svm.SVC(kernel='linear') 
    model.fit(train_vectors, train_labels)
    model.predict(test_vectors)
    ```
* text matching/similarity
  - Phonetic Matching
    ```
    import fuzzy 
    model = fuzzy.Soundex(4)
    model('word1')
    model('word2')
    ```
  - 