## ------- clustering -------
* pca with sklearn
  ```
  from sklearn.decomposition import PCA
  model = PCA(n_components=2).fit(X)
  model.explained_variance_ratio_
  model.singular_values_
  df = model.transform(X)
  ```
* sklearn clustering list [link](http://scikit-learn.org/stable/modules/clustering.html)
* kmeans
  - sklearn [link](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  ```
  from sklearn.cluster import KMeans
  from sklearn.metrics import silhouette_score
  KMeans(n_clusters=2, random_state=0).fit(df)
  kmeans.labels_
  plt.scatter(x,y,c=kmeans.labels_)
  round(silhouette_score(df,outcluters.labels_),2)
  ```
* MiniBatchKMeans
  ```
  from sklearn.cluster import MiniBatchKMeans
  outcluters = MiniBatchKMeans(n_clusters=3).fit(dfcluster)
  ```
* Birch
  ```
  from sklearn.cluster import Birch
  outcluters = Birch(n_clusters=3,threshold=0.2).fit(dfcluster)
  ```
* MeanShift
  ```
  from sklearn.cluster import MeanShift,estimate_bandwidth
  bandwidth = estimate_bandwidth(dfcluster, quantile=.4)
  outcluters = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(dfcluster)
  ```
* AffinityPropagation
  ```
  from sklearn.cluster import AffinityPropagation
  outcluters = AffinityPropagation(damping=0.9).fit(dfcluster)
  ```
* AgglomerativeClustering
  ```
  from sklearn.cluster import AgglomerativeClustering
  outcluters = AgglomerativeClustering(n_clusters=3).fit(dfcluster)
  ```
* SpectralClustering
  ```
  from sklearn.cluster import SpectralClustering
  outcluters = SpectralClustering(n_clusters=3,eigen_solver='arpack',affinity="nearest_neighbors").fit(dfcluster)
  ```
* DBSCAN
  ```
  from sklearn.cluster import DBSCAN
  outcluters = DBSCAN(eps=dfcluster[df_features[1]].median()).fit(dfcluster)
  ```
* GaussianMixture
  ```
  from sklearn.mixture import GaussianMixture
  outcluters = GaussianMixture(n_components=3,covariance_type='full').fit(dfcluster)
  ```
* nearest neighbors
  - sklearn
    ```
    from sklearn.neighbors import NearestNeighbors
    model = NearestNeighbors(n_neighbors=1).fit(X)
    model.kneighbors(xtest) #second array provide index in X which closest to xtest
    ```




## ------- Regression -------
* linear regression [link](http://scikit-learn.org/stable/modules/linear_model.html)
  - linear regression
  ```
  from sklearn.linear_model import LinearRegression
  reg = LinearRegression()
  reg.fit(df_train)
  reg.predict(df_test)
  reg.score(X,y) # r square
  ```
  - Ridge regression: Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of coefficients. 
  `reg = linear_model.Ridge (alpha = .5)`
  - Lasso regression:The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer parameter values, effectively reducing the number of variables upon which the given solution is dependent. 
  `reg = linear_model.Lasso(alpha = 0.1)`
  - regression with lasso regularization
    ```
    import sklearn.linear_model.Lasso
    model = Lasso().fit(X,y)
    model.predict(Xtest)
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
  pd.DataFrame({'value':clf.feature_importances_,'col':features.columns}).sort_values('value',ascending=False)
  ```
* lightgbm [doc](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)
  - sample 1
  ```
  import lightgbm as lgb
  param = {'num_leaves':31, 'num_trees':100, 'objective':'binary',
    'metric':['auc', 'binary_logloss']}
  model = lgb.train(param, dftrain,num_round=10,valid_sets=[dftest])
  model.predict(dftest)
  ```
  - sample 2
  ```
  import lightgbm as lgb
  param = {
    'boosting_type':'gbdt',
    'objective': 'regression',
    'nthread': 4,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'max_depth': -1,
    'subsample': 0.8,
    'bagging_fraction' : 1,
    'max_bin' : 5000 ,
    'bagging_freq': 20,
    'colsample_bytree': 0.6,
    'metric': 'rmse',
    'min_split_gain': 0.5,
    'min_child_weight': 1,
    'min_child_samples': 10,
    'scale_pos_weight':1,
    'zero_as_missing': True,
    'seed':0,
    'num_rounds':50000
  }
  train_set = lgbm.Dataset(x_train, y_train, silent=False,categorical_feature=['year','month'])
  valid_set = lgbm.Dataset(x_test, y_test, silent=False,categorical_feature=['year','month'])
  model = lgbm.train(params, train_set = train_set, num_boost_round=10000,early_stopping_rounds=500,verbose_eval=500, valid_sets=valid_set)
  model.predict(dftest)
  ```
* random forest [link](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
  ```
  from sklearn.ensemble import RandomForestRegressor
  regr = RandomForestRegressor(n_estimators=10 ,max_depth=2, random_state=0, n_jobs=-1)
  regr.fit(features, label)
  regr.feature_importances_
  pd.DataFrame({'value':regr.feature_importances_,'col':features.columns}).sort_values('value',ascending=False)
  ```
* xgboost
  - print rmse in each round
  ```
  import xgboost as xgb
  dftrain = xgb.DMatrix(X_train,label=y_train)
  dftest = xgb.DMatrix(X_test,label=y_test)

  model = xgb.train(
      params={'objective':'reg:linear','eval_metric':'rmse'},
      dtrain=dftrain, num_boost_round=20,
      early_stopping_rounds=10, evals=[(dftrain,'test')],)

  model.predict(xgb.DMatrix(dftest),
      ntree_limit = model.best_ntree_limit)
  ```
  - no printing
  ```
  import xgboost as xgb
  model = xgb.XGBRegressor(
    objective ='reg:linear',eval_metric='rmse',learning_rate = 0.1,
    num_boost_round=20,early_stopping_rounds=10)
  model.fit(X_train,y_train)
  prediction = model.predict(X_test)
  ```
* regression feature selection with p value
  ```
  from sklearn.feature_selection import f_regression
  f_regression(X_train,y_train)[1]
  ```




## ------- classification -------
* random forest
  - sklearn
    ```
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_jobs=-1)
    ```
    + n_jobs: 1 run 1 tree each time, -1 run all trees parallel
    + max_depth: None till min leaf, 4 num of leaves
    + n_estimators: 10 trees in forest
* naive bayes
  - GaussianNB
    ```
    from sklearn.naive_bayes import GaussianNB
    GaussianNB().fit(X,y)
    ```
* SVM
  - sklearn
    ```
    from sklearn.svm import SVC
    model = SVC(kernel='linear') #poly,rbf,sigmoid etc
    model.fit(X,y)
    model.predict(Xtest)
    ```
    + kernel: function is to convert raw data into another format. let problem from non-linear question to linear question.
    + c: control tradeoff between smooth/straight or non-smooth boundary. higher - non-smooth (possible over-fitting)
    + gamma:
* GradientBoostingClassifier[link](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)
  ```
  from sklearn.ensemble import GradientBoostingClassifier
  model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X_train, y_train)
  model.score(X_test, y_test)
  ```
* logistic regression
  ```
  from sklearn.linear_model import LogisticRegression
  reg = LogisticRegression(random_state=1)
  ```


## ------- Time Series -------
### moving average
* pandas pkg
```
import pandas as pd
pd.rolling_mean(df.col1, avg_length)
```

### holt-winters
* knowledge of method
actually it is triple exponential smoothing
* fecon235 pkg [link](https://github.com/rsvp/fecon235#dt_2015-08-01_094628)
* seasonal pkg [link](https://github.com/welch/seasonal/blob/master/examples/hw.py)
* pycast pkg
`pycast.methods.exponentialsmoothing.HoltWintersMethod`

### auto regression
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

### prophet package









## ------- Natural Language Processing -------
* intro with python [link](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
* open source book [Natural Language Processing with Python](http://www.nltk.org/book_1ed/)
* install packages
  - `pip install nltk` The complete toolkit for all NLP techniques.
  - `pip install pattern` A web mining module for the with tools for NLP and machine learning.
  - `pip install TextBlob` Easy to use nl p tools API, built on top of NLTK and Pattern.
  - `pip install Gensim` Topic Modelling for Humans
### explore function
* word only appear once
  `FreqDist(text1).hapaxes()`
* frequency distribution
  - all words
    `FreqDist(text1)`
  - single word
    `FreqDist(text1)['whale']`
  - plot word distribution
    `FreqDist(text1).plot(50,cumulative=True)`
* find words position  [case senstive]
  `text4.dispersion_plot(['citizens','democracy','freedom','duties','america'])`
* find words usually come along with
  `text2.common_contexts(['monstrous','very'])`
* find words in text
  `textstring.concordance('word')`
* find similar words
  `textstring.similar('word')`


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
    WordNetLemmatizer().lemmatize('word')
    ## 'multiply'
    ```
  - Stemming
    ```
    from nltk.stem.porter import PorterStemmer
    PorterStemmer().stem('multiplying')
    ## 'multipli'
    ```
* entity extraction with nltk
  - speech tagging
    ```
    from nltk import word_tokenize, pos_tag
    pos_tag(word_tokenize(a_sentence))
    ```
  - name entity
    ```
    from nltk import ne_chunk, pos_tag, word_tokenize
    from nltk.tree import Tree
    out = ne_chunk(pos_tag(word_tokenize(text)))
    [w for w in out if type(w)==Tree]
    ```
* entity extraction
  - Latent Dirichlet Allocation (LDA)
    ```
    # convert str to matrix
    import corpora
    dictionary = corpora.Dictionary(str.split())
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in str.split()]
    ```
  - topic features
    ```
    # train matrix to get topics
    from gensim import models
    models.ldamodel.LdaModel(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50).print_topics()
    ```
* statistic features/vectorization
  - count vectorization [sklearn doc](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.get_feature_names)
    ```
    from sklearn.feature_extraction.text import CountVectorizer
    out = CountVectorizer(stop_words='english')
    out.fit_transform(df[textcol]) #convert textcol to many columns and each col is one word
    out = CountVectorizer(analyzer=afun)
    out.fit_transform(df[textcol]) # apply customized functions before fit
    out.get_feature_names() #give all unique words
    # output will be sparse matrix so need to convert to df
    df = pd.DataFrame(out.toarray())
    ```
  - n-grams
    ```
    from sklearn.feature_extraction.text import CountVectorizer
    out = CountVectorizer(ngram_range=(1,1)).fit_transform(df[textcol]) #default 1 word 1 column
    out = CountVectorizer(ngram_range=(1,2)).fit_transform(df[textcol]) #1gram + bigrams
    out = CountVectorizer(ngram_range=(2,2)).fit_transform(df[textcol]) #bigrams
    # output will be sparse matrix so need to convert to df
    df = pd.DataFrame(out.toarray())
    # list of word vector combination
    out.get_feature_names()
    ```
  - Term Frequency – Inverse Document Frequency (TF – IDF)
    ```
    from sklearn.feature_extraction.text import TfidfVectorizer
    out = TfidfVectorizer().fit_transform(['a sentence', 'b sentence'])
    out = TfidfVectorizer(sbulinear_tf=True,max_df=0.5,stop_words='english').fit_transform(['a sentence', 'b sentence'])
    # remove stop words with tf_idf, and only keep 50% common words
    out = TfidfVectorizer(analyzer=customized_fun).fit_transform(df[textcol]) # customized_fun can do some cleaning or transformation before vectorized
    # output will be sparse matrix so need to convert to df
    df = pd.DataFrame(out.toarray())
    ```
* only choose most import x% features
  ```
  from sklearn.feature_selection import SelectPercentile,f_classif
  model = SelectPercentile(f_classif,percentile=x)
  model.fit(X,y).transform(X).toarray()
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
  - sklearn (random forest)
    ```
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators,max_depth)
    model.fit(X,y)
    model.predict(Xtest)
    ```
* text matching/similarity
  - Phonetic Matching
    ```
    import fuzzy 
    model = fuzzy.Soundex(4)
    model('word1')
    model('word2')
    ```





## ---- recommendation system ----
* find correlation with given series
  - `df.corrwith(df2)` df2 has only 1 series/column
* utility matrix
  - turn rating df into matrix with fill na with 0`df.pivot_table(values='rating',index='userid',columns='itemsid',fill_value=0)`
* SVD (reduce #cols)
  ```
  from sklearn.decomposition import TruncatedSVD
  TruncatedSVD(n_components=12,random_state=17).fit_transform(df)
  ```
*
*