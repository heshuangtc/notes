## ------- classification algorithm -------

* Decision Tree
  - works for both categorical and continuous dependent variables. split the population into two or more homogeneous sets. To split the population into different heterogeneous groups, it uses various techniques like Gini, Information Gain, Chi-square, entropy.
  - This is done based on most significant attributes/ independent variables to make as distinct groups as possible.
    + information gain: `entropy(parent)- weight_avg*entropy(children)`. entropy = `sum(p*log2(p))` and if children only has 1 class(1 p), entropy=0. bigger better. [udacity](https://classroom.udacity.com/courses/ud120/lessons/2258728540/concepts/24033885800923)

* SVM (Support Vector Machine)
  - plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate.
  -  the line such that the distances from the closest point in each of the two groups will be farthest away. This line is our classifier. 

* Naive Bayes
  - a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
  - Even if these features depend on each other or upon the existence of the other features, a naive Bayes classifier would consider all of these properties to independently contribute to the probability.
  - Naive Bayesian model is easy to build and particularly useful for very large data sets.
  - `P(c|x)` is the posterior probability of class (target) given predictor (attribute) / `P(c)` the prior probability of class. / `P(x|c)` the likelihood which is the probability of predictor given class.  / `P(x)` the prior probability of predictor. ![ml_terms-Bayes_rule](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/ml_terms-Bayes_rule.png)
    + multinomial NB: discrete frequency counts, categorical or continuous or binary
    + bernoulli NB: continuous variables
    + gaussian NB: feature are normal distribution

* kNN (k- Nearest Neighbors)
  - can be used for both classification and regression problems
  - stores all available cases and classifies new cases by a majority vote of its k neighbors. The case being assigned to the class is most common amongst its K nearest neighbors measured by a distance function.
  - Things to consider before selecting kNN:
    + KNN is computationally expensive
    + Variables should be normalized else higher range variables can bias it
    + Works on pre-processing stage more before going for kNN like outlier, noise removal




## ------- Ensemble learning -------
* ensemble method [wiki link](https://en.wikipedia.org/wiki/Ensemble_learning)
  - create multiple models and then combine them to produce better results
  - Gradient Boosting (boosting) / RandomForest (bagging)
  - boosting: increase weight based on previous error. iterative, slower. weighted voting. easier to overfit.
  - bagging: random samples. independent ,parallel. non-weighted voting.

* Random Forest [wiki](https://en.wikipedia.org/wiki/Random_forest)
  - each tree gives a classification/prediction (1 vote) independently and the forest chooses the classification with the most votes (over all the trees in the forest).
  - accept almost all types of inputs including class, continuous, missing values etc.
  - can be used for both classification and regression.

### Gradient Boosting Algorithms
* some links: [blog](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/) and [boost blog](https://www.analyticsvidhya.com/blog/2015/05/boosting-algorithms-simplified/)
* defination:
  - an iterative approach to combine weak learners to create strong learners by focusing on mistakes/error of prior iterations, heavy weight on error observation and then minimize the error.
  - accept almost all types of inputs including class, continuous, missing values etc.
  - GB also use decision tree.
  - can be used for both classification and regression.
* AdaBoost
* GBM
  - an ensemble of learning algorithms which combines the prediction of several base estimators in order to improve robustness over a single estimator. It combines multiple weak or average predictors to a build strong predictor.
* XGBoost
  - The XGBoost has an immensely high predictive power which makes it the best choice for accuracy in events as it possesses both linear model and the tree learning algorithm, making the algorithm almost 10x faster than existing gradient booster techniques.
  -  it is also called a regularized boosting technique. This helps to reduce over-fit modeling.
* LightGBM
  - uses tree based learning algorithms
  - it splits the tree leaf wise with the best fit BUT other boosting algorithms split the tree depth wise or level wise rather than leaf-wise.
  - Faster training speed and higher efficiency / Lower memory usage / Better accuracy / Parallel and GPU learning supported / Capable of handling large-scale data
* Catboost
  - easily integrate with deep learning frameworks like Google’s TensorFlow and Apple’s Core ML
  - handle missing data well before you proceed with the implementation
  - can automatically deal with categorical variables without showing the type conversion error



## ------- Neural Networks -------
* [Google NN course](https://developers.google.com/machine-learning/crash-course/introduction-to-neural-networks/anatomy)
* neural networks might help with nonlinear problems
* Hidden Layers
  - a "hidden layer" of intermediary values
  - Each yellow node in the hidden layer is a weighted sum of the blue input node values.
  - The output is a weighted sum of the yellow nodes.
  - Is this model linear? Yes—its output is still a linear combination of its inputs![hidden layers](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-ml_nn_hiddenlayers.png)
* Activation Functions
  - To model a nonlinear problem, we can pipe each hidden layer node through a nonlinear function. This nonlinear function is called the activation function.![hidden layers](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-ml_nn_activationfunctions.png)
  - Common Activation Functions
    + sigmoid: converts the weighted sum to a value between 0 and 1 `1/(1+e^-x)`
    + rectified linear unit: often works a little better than a smooth function like the sigmoid, while also being significantly easier to compute.`max(0,x)`
    + any mathematical function can serve as an activation function
  -  the value of a node in the network is given by the following formula: `Fun_activation(w*x+b)`

  >>A caveat: neural networks aren't necessarily always better than feature crosses, but neural networks do offer a flexible alternative that works well in many cases.

* Failure Cases[link](https://developers.google.com/machine-learning/crash-course/training-neural-networks/best-practices)
  - Vanishing Gradients : for the lower layers (closer to the input) can become very small. When the gradients vanish toward 0 for the lower layers, these layers train very slowly.
    + The ReLU activation function can help prevent vanishing gradients.
  - Exploding Gradients : If the weights in a network are very large, then the gradients for the lower layers involve products of many large terms. 
    + Batch normalization can help prevent exploding gradients, as can lowering the learning rate.
  - Dead ReLU Units: Once the weighted sum for a ReLU unit falls below 0, the ReLU unit can get stuck. 
    + Lowering the learning rate can help keep ReLU units from dying.

* Dropout Regularization
  - Yet another form of regularization, called Dropout, is useful for neural networks.  The more you drop out, the stronger the regularization.
  - 0.0 = No dropout regularization.
  - 1.0 = Drop out everything. The model learns nothing.
  - Values between 0.0 and 1.0 = More useful.

### Multi-Class Neural Networks [link](https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/one-vs-all)
* One vs. all 
  - provides a way to leverage binary classification. Given a classification problem with N possible solutions, a one-vs.-all solution consists of N separate binary classifiers—one binary classifier for each possible outcome.
* Softmax
  - Softmax assigns decimal probabilities to each class in a multi-class problem. Those decimal probabilities must add up to 1.0. 
  - This additional constraint helps training converge more quickly than it otherwise would.
  - Softmax is implemented through a neural network layer just before the output layer. The Softmax layer must have the same number of nodes as the output layer.
  - Softmax Options
    + Full Softmax is the Softmax we've been discussing; that is, Softmax calculates a probability for every possible class.
    + Candidate sampling means that Softmax calculates a probability for all the positive labels but only for a random sample of negative labels.
  >> Full Softmax is fairly cheap when the number of classes is small but becomes prohibitively expensive when the number of classes climbs. Candidate sampling can improve efficiency in problems having a large number of classes.
  - One Label vs. Many Labels: Softmax assumes that each example is a member of exactly one class. Some examples, however, can simultaneously be a member of multiple classes.

### Embeddings
* embeddings: translate large sparse vectors into a lower-dimensional space that preserves semantic relationships. 
* Collaborative Filtering[link](https://developers.google.com/machine-learning/crash-course/embeddings/motivation-from-collaborative-filtering)
  - the task of making predictions about the interests of a user based on interests of many other users.
  - one dimension/ two dimension
  -  each such dimension is called a latent dimension, as it represents a feature that is not explicit in the data but rather inferred from it.
* Categorical Input Data
  - input features that represent one or more discrete items from a finite set of choices.
  - Categorical data is most efficiently represented via sparse tensors, which are tensors with very few non-zero elements.
* Size of Network
  - Amount of data. The more weights in your model, the more data you need to train effectively.
  - Amount of computation. The more weights, the more computation required to train and use the model. It's easy to exceed the capabilities of your hardware.
* one-hot encoding
  -  because only one index has a non-zero value.
* Translating to a Lower-Dimensional Space
  - solve the core problems of sparse input data by mapping your high-dimensional data into a lower-dimensional space.
* Embeddings as lookup tables: An embedding is a matrix in which each column is the vector that corresponds to an item in your vocabulary. 
* Obtaining Embeddings[link](https://developers.google.com/machine-learning/crash-course/embeddings/obtaining-embeddings)
  -  principal component analysis (PCA):  find highly correlated dimensions that can be collapsed into a single dimension.
  -  Word2vec: an algorithm invented at Google for training word embeddings. Word2vec relies on the distributional hypothesis to map semantically similar words to geometrically close embedding vectors![hidden layers](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-ml_embedding.png)