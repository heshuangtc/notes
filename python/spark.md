* Spark
    - run in clusters servers
    - install
        + from apache spark site
        + require java installed
        + spark-shell for scala, pyspark for python, sparkR for R in bin

## load/create
* load a file
    ```
    df = spark.read.csv('path/file.csv',header=True)
    ```
* display basic info

    - `df.schema`
    - `df.printSchema()`
    - `df.columns`
    - all df `df.show()`
    - top 5 rows `df.take(5)`
    - num of rows `df.count()`

## subset/manipulate
* subset df
    - sample of original df 10% and not replace original df
    `df = df.sample(False,0.1)`
    - keep rows where col1 is greater than 10
    `df2 = df.filter('col1>10')`
    - select col1 from df
    ```
    df.select('col1').show()
    df.select('col1','col2').show()
    ```
* create a data frame
    - with num input data
        ```
        from pyspark.ml.linalg import Vectors
        df = spark.createDataFrame([
                (1,Vectors.dense([10.0,10000,1.0]),),
                (2,Vectors.dense([20.0,30000,2.0]),),
                (3,Vectors.dense([30.0,40000,3.0]),)
                ], ['id','features'])
        ```
    - with text input data
        ```
        from pyspark.ml.linalg import Vectors
        df = spark.createDataFrame([
                (1,'this is a sentence'),
                (2,'another one'),
                (3,'last one')
                ], ['id','text'])
        ```
    - with a list (only 1 col)
        ```
        df = spark.createDataFrame([(2),(2),(5),(9)],['features'])
        ```
* change col name
    ```
    df.select(col('colname1').alias('newcolname1'),
              col('colname2').alias('newcolname2'))
    ```
* change few columns into 1 vector col
    ```
    from pyspark.ml.feature import VectorAssembler
    dfout = VectorAssembler(inputCol=['col1','col2','col3'],outputCol='newcol').transform(df)
    ```
* convert string/categorical col into index/numeric col
    ```
    from pyspark.ml.feature import StringIndexer
    dfout = StringIndexer(inputCol='col',outputCol='newcol').fit(df).transform(df)
    ```
* `from pyspark.sql.functions import *`
* split data based on percentage
    ```
    df0 = df.randomSplit[(0.6,0.4),seed = 1]
    df6 = df0[0]
    df4 = df0[1]
    ```
*

## preporcessing
    
* normalization/scaling
    - MinMax (0 to 1)
    ```
    from pyspark.ml.feature import MinMaxScaler
    model = MinMaxScaler(inputCol='col',outputCol='scaled_col')
    model.fit(df)
    dfout = model.transform(df)
    ```

* standardization
    - normal distribution (-1 to 1)
    ```
    from pyspark.ml.feature import StandardScaler
    model = StandardScaler(inputCol='col',outputCol='scaled_col',withStd=True, withMean=True)
    model.fit(df)
    dfout = model.transform(df)
    ```

* group data by boundary/bins
    ```
    from pyspark.ml.feature import Bucketizer
    bin_splits = [-float('inf'),-10,0,10,float('inf')]
    model = Bucketizer(splits = bin_splits,inputCol='col',outputCol='outcol')
    dfout = model.transform(df)
    ```
* split string into words
    ```
    from pyspark.ml.feature import Tokenizer
    model = Tokenizer(inputCol='textcol',outputCol='words')
    dfout = model.transform(df)
    ```
* vectorization words
    - TF-IDF
        ```
        from pyspark.ml.feature import HashingTF, IDF
        model = HashingTF(inputCol='wordscol',outputCol='rawfeature',numFeatures=20) 
        # inputCol needs words list instead of entire string
        # numFeatures: how many features want to keep
        dfout = model.transform(df)
        model = IDF(inputCol='rawfeature',outputCol='idf_feature')
        moel.fit(dfout).transform(dfout)
        ```
*
*


## clustering
* k-means
    ```
    from pyspark.ml.clustering import KMeans
    model = KMeans().setK(3).setSeed(1) #set number of k =3
    model.fit(df) #df needs vector
    model.clusterCenters() #provide cluster centers
    ```
* BiscectingKMeans
    ```
    from pyspark.ml.clustering import BiscectingKMeans
    model = BiscectingKMeans().setK(3).setSeed(1) #set number of k =3
    model.fit(df) #df needs vector
    model.clusterCenters() #provide cluster centers
    ```
*
*


## classification
* Naive Bayes
    ```
    from pyspark.ml.classification import NaiveBayes
    model = NaiveBayes(modelType='multinomial').fit(dftrain)
    dfprediction = model.transform(dftest) #default outcol is prediction
    ```
* Neural Network
    ```
    from pyspark.ml.classification import MultilayerPerceptronClassifier
    layers = [4,5,5,3] #number of nodes in each layers
    model = MultilayerPerceptronClassifier(layers=layers,seed=1)
    model.fit(dftrain)
    prediction = model.transform(dftest)
    ```
    - first layer: num of nodes are number of input features
    - middle layer: num of nodes can be any number
    - last layer: num of nodes must be num of categories of targetcol
* Decision Tree
    ```
    from pyspark.ml.classification import DecisionTreeClassifier
    model = DecisionTreeClassifier(labelCol='labelCol',featuresCol='vectorfeature_col')
    model.fit(dftrain)
    prediction = model.transform(dftest)
    ```
*
*
*
*
## regression
* linear regression
    ```
    from pyspark.ml.regression import LinearRegression
    model = LinearRegression(featuresCol='vectorfeature_col',labelCol='targetcol')
    model.fit(dftrain)

    # display model attributes
    model.coefficients
    model.intercept
    model.summary.rootMeanSquaredError
    model.save('xxx.model')
    ```
* decision tree regression
    ```
    from pyspark.ml.regression import DecisionTreeReggressor
    model = DecisionTreeReggressor(featuresCol='vectorfeature_col',labelCol='targetcol')
    model.fit(dftrain)
    prediction = model.transform(dftest)
    ```
* gradient-boosted tree regression
    ```
    from pyspark.ml.regression import GBTRegressor
    model = GBTRegressor(featuresCol='vectorfeature_col',labelCol='targetcol')
    model.fit(dftrain)
    prediction = model.transform(dftest)
    ```
*
*
*
*
*
## evaluation
* MulticlassClassificationEvaluator
```
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
MulticlassClassificationEvaluator(labelCol='targetcol',predictionCol='prediction',metricName='accuracy').evaluate(dfprediction)
MulticlassClassificationEvaluator(metricName='accuracy').evaluate(dfprediction)
```

* RegressionEvaluator
    ```
    from pyspark.ml.evaluation import RegressionEvaluator
    RegressionEvaluator(labelCol='labelCol',predictionCol='prediction',metricName='rmse')
    ```
*
*
*
## recommendation system
* collaborative filtering
```
```
* content-based
```
```
*