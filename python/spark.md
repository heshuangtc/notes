* Spark
    - run in clusters servers
    - install
        + from apache spark site
        + require java installed
        + spark-shell for scala, pyspark for python, sparkR for R in bin

## load/create/subset
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


## clustering
*
*
*
*
*
*
*
*
*
*
*
*