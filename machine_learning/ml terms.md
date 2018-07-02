## ------- regression algorithm -------

from blog [link](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)

* linear regression
  - estimate real values based on continuous variable(s). establish relationship between independent and dependent variables by fitting a best line. This best fit line is known as regression line and represented by a linear equation `Y= a*X + b`.
  - These coefficients a and b are derived based on minimizing the sum of squared difference of distance between data points and regression line.
  - `Y` Dependent Variable / `a` Slope / `X` Independent variable / `b` Intercept
  - Simple Linear Regression is characterized by one independent variable. And, Multiple Linear Regression is characterized by multiple independent variables.

* logistic regression [link](https://developers.google.com/machine-learning/crash-course/logistic-regression/calculating-a-probability)
  -  a classification not a regression algorithm. returns a probability and need classification threshold (also called the decision threshold).
  -  estimate discrete values ( Binary values like 0/1, yes/no, true/false ) based on given set of independent variable(s). In simple words, it predicts the probability of occurrence of an event by fitting data to a logit function (a sigmoid function to make sure output always falls between 0 and 1).
  -  the log odds of the outcome is modeled as a linear combination of the predictor variables.
  ```
    odds= p/ (1-p) = probability of event occurrence / probability of not event occurrence
    ln(odds) = ln(p/(1-p))
    logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk
  ```
  -  p is the probability of presence of the characteristic of interest. It chooses parameters that maximize the likelihood of observing the sample values rather than that minimize the sum of squared errors (like in ordinary regression).


## ------- classification algorithm -------
* Decision Tree
  - works for both categorical and continuous dependent variables. split the population into two or more homogeneous sets. To split the population into different heterogeneous groups, it uses various techniques like Gini, Information Gain, Chi-square, entropy.
  - This is done based on most significant attributes/ independent variables to make as distinct groups as possible.

* SVM (Support Vector Machine)
  - plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate.
  -  the line such that the distances from the closest point in each of the two groups will be farthest away. This line is our classifier. 

* Naive Bayes
  - a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
  - Even if these features depend on each other or upon the existence of the other features, a naive Bayes classifier would consider all of these properties to independently contribute to the probability.
  - Naive Bayesian model is easy to build and particularly useful for very large data sets.
  - `P(c|x)` is the posterior probability of class (target) given predictor (attribute) / `P(c)` the prior probability of class. / `P(x|c)` the likelihood which is the probability of predictor given class.  / `P(x)` the prior probability of predictor. ![ml_terms-Bayes_rule](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/ml_terms-Bayes_rule.png)

* kNN (k- Nearest Neighbors)
  - can be used for both classification and regression problems
  - stores all available cases and classifies new cases by a majority vote of its k neighbors. The case being assigned to the class is most common amongst its K nearest neighbors measured by a distance function.
  - Things to consider before selecting kNN:
    + KNN is computationally expensive
    + Variables should be normalized else higher range variables can bias it
    + Works on pre-processing stage more before going for kNN like outlier, noise removal

* Random Forest [wiki](https://en.wikipedia.org/wiki/Random_forest)
  - To classify a new object based on attributes, each tree gives a classification and we say the tree “votes” for that class. The forest chooses the classification having the most votes (over all the trees in the forest).

### Neural Networks [link](https://developers.google.com/machine-learning/crash-course/introduction-to-neural-networks/anatomy)
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



## ------- clustering algorithms -------
* sample data comparison with python[link](http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py)![cluster_comparison](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-clustering.png)

* K-Means [wiki](https://en.wikipedia.org/wiki/K-means_clustering)
  - unsupervised algorithm which  solves the clustering problem
  - Given a set of observations (x1, x2, …, xn), where each observation is a d-dimensional real vector, k-means clustering aims to partition the n observations into k (≤ n) sets S = {S1, S2, …, Sk} so as to minimize the within-cluster sum of squares (WCSS). ![wcss](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-ml_term-kmeans-wcss.png)
  - Because the total variance is constant, this is also equivalent to maximizing the squared deviations between points in different clusters (between-cluster sum of squares, BCSS).![bcss](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-ml_term-kmeans-bcss.png)
  - advantage: fast, few computing
  - disadvantage: unknown num of center, center is random located(clusters may differ in each run)
* K-Median
  - similar with K-Means instead of recomputing the group center points using the median. 

* Mean-Shift Clustering[link](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68)
  -  a sliding-window-based algorithm that attempts to find dense areas of data points.many window filters and move towards higher density.
  - advantage: dont have to set num of centers, outliers as noises but included
  - disadvantage: window size r is unknown

* Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
  - a density based clustered algorithm similar to mean-shift
  - with e distance moving center, data points are in one cluster. iteration till all data processed.
  - advantage: num center is unknown, outliers as noises and ignored
  - disadvantage: not good for clusters are of varying density, e is unknown

* Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)
  - GMMs assume the data points are Gaussian distributed
  - two parameters to describe the shape of the clusters: the mean and the standard deviation
  - use an optimization algorithm called Expectation–Maximization (EM) to find mean and sd
  - compute the probability that each data point belongs to a particular cluster. maximize the probabilities of data points within the clusters
  - K-Means is actually a special case of GMM in which each cluster’s covariance along all dimensions approaches 0.
  - advantage: good for any ellipse shape, Secondly, since GMMs use probabilities, they can have multiple clusters per data point.

* Agglomerative Hierarchical Clustering
  - Hierarchical clustering algorithms actually fall into 2 categories: top-down or bottom-up.
  - Bottom-up hierarchical clustering is therefore called hierarchical agglomerative clustering or HAC.
  - each data point as a single cluster, select a distance(average linkage) metric that measures the distance between two clusters, combined 2 clusters with the smallest average linkage.


* Dimensionality Reduction Algorithms
  - identify highly significant variable(s)
  - dimensionality reduction algorithm helps us along with various other algorithms like Decision Tree, Random Forest, PCA, Factor Analysis, Identify based on correlation matrix, missing value ratio and others.



### Gradient Boosting Algorithms

from blog [link](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/) and boost blog [link](https://www.analyticsvidhya.com/blog/2015/05/boosting-algorithms-simplified/)

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
 










## ------- Ensemble learning -------
* wiki link: https://en.wikipedia.org/wiki/Ensemble_learning

### representation learning

* autoencoder



## ------- validation -------

### logistic regression
* confusion matrix(for binary classification)
  - confusion table/confusion matrix

|..|Actual True|Actual False|
|-------|-------|-------|
|Predict postive|True Positive (TP)|False Positive (FP) type I error|
|Predict negative|False Negative (FN) type II error|True Negative (TN)|

  - accuracy = `#correct predictions/total # predictions` = `(TP+TN)/(TP+TN+FP+FN)`
  - precision = `TP/(TP+FP)` What proportion of positive identifications was actually correct?
  - recall = `TP/(TP+FN)` What proportion of actual positives was identified correctly?

* ROC curve
  -  ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds
  -  plot: x,y is FPR,TPR with different thresholds
  -  True Positive Rate = recall = `TP/(TP+FN)`
  -  False Positive Rate = `FP/(FP+TN)`
* AUC
  - Area Under the ROC Curve
  - AUC provides an aggregate measure of performance across all possible classification thresholds.
  - AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.
  - AUC is scale-invariant. It measures how well predictions are ranked, rather than their absolute values.Scale invariance is not always desirable. 
  - AUC is classification-threshold-invariant. It measures the quality of the model's predictions irrespective of what classification threshold is chosen. Classification-threshold invariance is not always desirable.

* prediction bias
  - Logistic regression predictions should be unbiased. `average of predictions should ≈ average of observations`
  - Prediction bias = `average of predictions - average of labels in dataset`
  - Possible root causes of prediction bias are:
    + Incomplete feature set
    + Noisy data set
    + Buggy pipeline
    + Biased training sample
    + Overly strong regularization

#### regularization
* generalization curve, which shows the loss for both the training set and validation set against the number of training iterations
* loss: measures how well the model fits the data
* regularization: measures model complexity. 
  - it is to prevent over-fitting by penalizing complex models
  - minimize loss (empirical risk minimization):min(Loss(Data|Model))
  - structural risk minimization: min(Loss(Data|Model)+lambda*complexity(Model))
    + Increasing the lambda value strengthens the regularization effect.
    + Lowering the value of lambda tends to yield a flatter histogram

* Model complexity
  - a function of the weights of all the features in the model
    + defines the regularization term as the sum of the squares of all the feature weights
    + L2 regularization = sum(Wi^2)
    + weights close to zero have little effect on model complexity, while outliers weights can have a huge impact
  - a function of the total number of features with nonzero weights.

* L1 regularization[link](https://developers.google.com/machine-learning/crash-course/regularization-for-sparsity/l1-regularization)
  - problem: Given such high-dimensional feature vectors, model size may become huge and require huge amounts of RAM.
  - solution: 
    + A weight of exactly 0 essentially removes the corresponding feature from the model. Zeroing out features will save RAM and may reduce noise in the model.
    + Unfortunately not. L2 regularization encourages weights to be small, but doesn't force them to exactly 0.0.
    + use L1 regularization to encourage many of the uninformative coefficients in our model to be exactly 0, and thus reap RAM savings at inference time.

* L1 vs L2
|L2|L1|
|----|----|
|L2 penalizes weight2|L1 penalizes |weight||
|The derivative of L2 is 2 * weight|The derivative of L1 is k (a constant, whose value is independent of weight)|
|removes x% of the weight every time|subtracts some constant from the weight every time|





## ------- other -------
alphago zero nature article [link](https://www.nature.com/nature/journal/v550/n7676/full/nature24270.html)

field-aware factorization machines

factorization machines
