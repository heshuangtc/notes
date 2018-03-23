## ------- machine learning -------
### supervised algorithm methods

from blog [link](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)

* linear regression
  - estimate real values based on continuous variable(s). establish relationship between independent and dependent variables by fitting a best line. This best fit line is known as regression line and represented by a linear equation `Y= a*X + b`.
  - These coefficients a and b are derived based on minimizing the sum of squared difference of distance between data points and regression line.
  - `Y` Dependent Variable / `a` Slope / `X` Independent variable / `b` Intercept
  - Simple Linear Regression is characterized by one independent variable. And, Multiple Linear Regression is characterized by multiple independent variables.
* logistic regression
  -  a classification not a regression algorithm
  -  estimate discrete values ( Binary values like 0/1, yes/no, true/false ) based on given set of independent variable(s). In simple words, it predicts the probability of occurrence of an event by fitting data to a logit function.
  -  the log odds of the outcome is modeled as a linear combination of the predictor variables.
  ```
    odds= p/ (1-p) = probability of event occurrence / probability of not event occurrence
    ln(odds) = ln(p/(1-p))
    logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk
  ```
  -  p is the probability of presence of the characteristic of interest. It chooses parameters that maximize the likelihood of observing the sample values rather than that minimize the sum of squared errors (like in ordinary regression).
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
* K-Means [wiki](https://en.wikipedia.org/wiki/K-means_clustering)
  - unsupervised algorithm which  solves the clustering problem
  - Given a set of observations (x1, x2, …, xn), where each observation is a d-dimensional real vector, k-means clustering aims to partition the n observations into k (≤ n) sets S = {S1, S2, …, Sk} so as to minimize the within-cluster sum of squares (WCSS). ![wcss](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-ml_term-kmeans-wcss.png)
  - Because the total variance is constant, this is also equivalent to maximizing the squared deviations between points in different clusters (between-cluster sum of squares, BCSS).![bcss](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-ml_term-kmeans-bcss.png)
* Random Forest [wiki](https://en.wikipedia.org/wiki/Random_forest)
  - To classify a new object based on attributes, each tree gives a classification and we say the tree “votes” for that class. The forest chooses the classification having the most votes (over all the trees in the forest).
  - 
* Dimensionality Reduction Algorithms

### Gradient Boosting Algorithms

from blog [link](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)

* GBM
* XGBoost
* LightGBM
* Catboost


### Ensemble learning
* wiki link: https://en.wikipedia.org/wiki/Ensemble_learning

### representation learning

* autoencoder

### other

alphago zero nature article [link](https://www.nature.com/nature/journal/v550/n7676/full/nature24270.html)

field-aware factorization machines

factorization machines

## ------- deep learning -------

* deep learning  [wiki](https://en.wikipedia.org/wiki/Deep_learning) [a_site](http://deeplearning.net/) [book:deeplearning](http://www.deeplearningbook.org/)

* feedforward deep network, or multilayer perceptron(MLP)

* CNN [beginner guide](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/) [tutorial](http://deeplearning.net/tutorial/lenet.html)

* RNN [intro](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

* LSTM 