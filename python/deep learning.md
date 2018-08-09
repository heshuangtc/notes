## TensorFlow
* get started [link](https://www.tensorflow.org/get_started/eager)
  - setup `pip install tensorflow`
  - upgrade `pip install -q --upgrade tensorflow`
  - Once eager execution is enabled, it cannot be disabled within the same program. `import tensorflow as tf;tf.enable_eager_execution()`
  - load csv from url `urlfile = tf.keras.utils.get_file(fname=os.path.basename(url),origin=url)`
  - Parse the dataset 
    ```
    def parse_csv(line):
      example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
      parsed_line = tf.decode_csv(line, example_defaults)
      # First 4 fields are features, combine into single tensor
      features = tf.reshape(parsed_line[:-1], shape=(4,))
      # Last field is the label
      label = tf.reshape(parsed_line[-1], shape=())
      return features, label
    df = tf.data.TextLineDataset(urlfile)  #load csv text file
    df = df.skip(1) #skip header line
    df = df.map(parse_csv)      # parse each row
    df = df.shuffle(buffer_size=1000)  # randomize
    df = df.batch(32)
    ```
  - create a keras NN model
    ```
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(3)
    ])
    ```
    2 Dense layers with 10 nodes each, and an output layer with 3 nodes representing our label predictions. 
  - fit model with gradient function and loss function
    ```
    def loss(model, x, y):
      y_ = model(x)
      return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

    def grad(model, inputs, targets):
      with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
      return tape.gradient(loss_value, model.variables)
    grad(model, x, y) # fit model
    ```
  - optimizer 
    ```
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grads = grad(model, x, y) # fit model
    optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
    ```

- tfe function
  ```
  import tensorflow.contrib.eager as tfe
  tfe.metrics.Mean() #mean function
  tfe.metrics.Accuracy() #accuracy function
  ```
  - train model `tf.argmax(model(x), axis=1, output_type=tf.int32)` x is in 
  

* predict
  - `tf.argmax(model(x), axis=1, output_type=tf.int32)` x is in test data
  - `model(tf.convert_to_tensor(predict_dataset))`

* optimization algorithms [link](https://www.tensorflow.org/api_guides/python/train)
  - stochastic gradient descent (SGD) `tf.train.GradientDescentOptimizer(learning_rate=0.01)`

* define feature column [link](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list)

https://www.tensorflow.org/get_started/get_started_for_beginners

* example
  - classification DNNClassifier
    ```
    from sklearn.metrics import classification_report
    import tensorflow as tf
    tf.enable_eager_execution()

    train_features = dftrain.drop('label',axis=1).copy()
    train_label = dftrain['label'].copy()  # notice this is series instead of data frame
    test_features = dftest.drop('label',axis=1).copy()
    test_label = dftest['label'].copy() # notice this is series instead of data frame if we need data frame, use df[['col']]

    input_fn_train = tf.estimator.inputs.pandas_input_fn(
            x=train_features,y=train_label, shuffle=True)
    input_fn_validation = tf.estimator.inputs.pandas_input_fn(
            x=train_features,y=train_label, shuffle=False)
    input_fn_test = tf.estimator.inputs.pandas_input_fn(
            x=test_features,y=test_label, shuffle=False)

    # if all columns are numeric continuous cols
    my_feature_columns = []
    for i in train_features.columns:
        my_feature_columns.append(tf.feature_column.numeric_column(key=i))

    classifier = tf.estimator.DNNClassifier(
            feature_columns=my_feature_columns,
            hidden_units=[20, 10, 10, 5],
            n_classes=5,
            optimizer=tf.train.AdagradOptimizer(learning_rate=0.5))
    classifier.train(input_fn=input_fn_train, steps=100)

    predictions_train = classifier.predict(input_fn=input_fn_validation)
    predictions_train = list(predictions_train)
    predictions_train = pd.DataFrame(predictions_train)['class_ids'].apply(lambda i:i[0])

    classification_report(train_label,predictions_train.predict)
    ```
  - 



## Keras

##





## image classification
* cat/dog kaggle CNN example with data augmentation and dropout [link](https://colab.research.google.com/github/google/eng-edu/blob/master/ml/pc/exercises/image_classification_part2.ipynb?utm_source=practicum-IC&utm_campaign=colab-external&utm_medium=referral&hl=en&utm_content=imageexercise2-colab#scrollTo=OpFqg-R1g9n6)