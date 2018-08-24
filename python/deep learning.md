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
* basic info
  - high level and based on either TensorFlow or Theoran
* reference
  - [keras doc](https://keras.io/layers/recurrent/#lstm)
  - [keras example](https://faroit.github.io/keras-docs/0.3.3/examples/#stacked-lstm-for-sequence-classification)
* first layer
  - if 2d data frame(normal number) `model.add(Dense(numofnodes, input_dim=numofcol, activation='relu'))`
  - 
* last layer
  - if continuous target variable `model.add(Dense(1, activation='linear'))`
  - if categorical target `model.add(Dense(numOFlabelclass, activation='softmax'))`
  - 
* layers
  - give name for a layer `model.add(Dense(1, activation='linear', name='xxx'))`
* model fit
  - epochs: too few - not enough accuracy, too much - waste time
  - shuffle: train better when True
  - verbose: =2 more detail printing when training
* load/save mode
  - save `model.save('name.h5')`
  - load `from keras.models import load_model;load_mode('name.h5)`
* logger
  - create a TensorBoard logger
    ```
    logger = keras.callbacks.TensorBoard(
      log_dir='logs', write_graph=True, histogram_freq=5
      )
    model.fit(....., callbacks=[logger])
    ```
  - display TensorBoard locally
    + open terminal
    + `tensorboard --logdir=path/logs`
    + copy url to browser

* LSTM (RNN)
  - input is dataframe. as data frame is 2d matrix, while LSTM needs 3d matrix, need to convert data frame format first. pay attention on input_shape in LSTM. output_dim is number of label class if multiple classes.
    ```
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.utils import to_categorical

    # convert data frame to 3d feature matrix and 2d target matrix
    x_train = np.array(df_features).reshape(NrowOfDf,1,NcolOfDf)
    y_train = np.array(df_label).reshape(NrowOfDf,1)
    # label columns is multi-categorical column, need to convert to dummy variables
    y_train = to_categorical(y_train) #will create numOFlabelclass columns

    # create model
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(1,NcolOfDf)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32, return_sequences=False)) 
    model.add(Dense(numOFlabelclass, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(x_train, y_train, batch_size=64, nb_epoch=5,validation_split=0.3)

    dfout = pd.DataFrame(model.predict(x_train))
    ```
  - 
* image (RNN)
  ```
  from keras.preporcessing.image import load_img, array_to_img
  from.utils.np_utils import to_categorical
  from keras.models import Sequential
  from keras.layers import Dense

  X_train.shape #(rows,xpix,ypix) 3d
  Y_train.shape #(rows,)

  # preprocessing image
  ## reshape data into 1 layer
  X_train.reshape(nrows,xpix*ypix)
  X_train.shape #(rows,xpix*ypix) 2d
  X_train = X_train.astype('float32')
  ## scale to 0,1
  X_train /=255 #color range is 0-255
  ## y label has 10 classes, categorical label to binary columns
  Y_train = to_categorical(Y_train,10)
  Y_train.shape #(rows,10)

  # model
  ## build
  model = Sequential()
  model.add(Dense(num_nodes, activation='relu', input_shape=(xpix*ypix,))) # first layer
  model.add(Dense(num_nodes, activation='relu'))
  model.add(Dense(10, activation='softmax')) # final layer
  ## compile
  model.compile(optimizer='adam', loss='cateorical_crossentropy', metrics=['accuracy']) #as y labels is multi-class category, so use this loss function
  model.summary()
  ## train
  model.fit(X_train,Y_train, epochs=20, validation_data=(X_train,Y_train))

  #plot
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.plot(history.history['loss'])

  # predict
  model.predict(X_test)
  ```

* image (CNN)
  ```
  from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
  from keras.models import Sequential
  from keras.utils import to_categorical

  X_train.shape #(rows,xpix,ypix) 3d
  Y_train.shape #(rows,)
  
  # preprocessing image
  ## reshape data
  X_train.reshape(nrows,xpix,ypix,1)
  X_train.shape #(rows,xpix,ypix,1) 4d
  X_train = X_train.astype('float32')
  ## scale to 0,1
  X_train /=255
  ## categorical label to binary columns
  Y_train = to_categorical(Y_train,num_label_class)
  Y_train.shape #(rows,num_label_class)

  # model
  ## build
  model = Sequential()
  model.add(Conv2D(32, kernel_size(5,5), input_shape=(xpix,ypix,1), pading='same', activation='relu'))
  model.add(MaxPooling2D()) #one conv layer needs one maxpool
  model.add(Conv2D(64, kernel_size(5,5), pading='same', activation='relu'))
  model.add(MaxPooling2D())
  model.add(Flatten()) # fully connection
  model.add(Dense(1024, activation='relu')) # fully connection
  model.add(Dense(num_label_class, activation='softmax')) #last layer
  ## compile
  model.compile(optimizer='adam', loss='cateorical_crossentropy', metrics=['accuracy'])
  ## train
  model.fit(X_train,Y_train, epochs=5, verbose=1, validation_data=(X_train,Y_train))
  model.evaluate(X_train, Y_train)
  ```
* image classification with exiting model
  ```
  import numpy as np
  from keras.preprocessing import image
  from keras.applications import resnet50

  model = resnet50.ResNet50()
  img = image.load_img("name.jpg", target_size=(224, 224))

  x = image.img_to_array(img)
  # Add a forth dimension since Keras expects a list of images
  x = np.expand_dims(x, axis=0)
  x = resnet50.preprocess_input(x)

  predictions = model.predict(x)
  predicted_classes = resnet50.decode_predictions(predictions, top=9)
  ```
##





## image classification
* cat/dog kaggle CNN example with data augmentation and dropout [link](https://colab.research.google.com/github/google/eng-edu/blob/master/ml/pc/exercises/image_classification_part2.ipynb?utm_source=practicum-IC&utm_campaign=colab-external&utm_medium=referral&hl=en&utm_content=imageexercise2-colab#scrollTo=OpFqg-R1g9n6)