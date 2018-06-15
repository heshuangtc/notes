## ------- data engineering -------
### feature engineering
*
* good features [link](https://developers.google.com/machine-learning/crash-course/representation/qualities-of-good-features)
  - Avoid rarely used discrete feature values
  - Prefer clear and obvious meanings
  - Don't mix "magic" values with actual data. To work around magic values, convert the feature into two features:One feature holds only quality ratings, never magic values. One feature holds a boolean value indicating whether or not.
  - The definition of a feature shouldn't change over time

### clean data [link](https://developers.google.com/machine-learning/crash-course/representation/cleaning-data)
* Scaling feature values

  Scaling means converting floating-point feature values from their natural range (for example, 100 to 900) into a standard range (for example, 0 to 1 or -1 to +1). 

  If a feature set consists of only a single feature, then scaling provides little to no practical benefit. If, however, a feature set consists of multiple features, then feature scaling provides the following benefits:

  - Helps gradient descent converge more quickly.
  - Helps avoid the "NaN trap," in which one number in the model becomes a NaN (e.g., when a value exceeds the floating-point precision limit during training), and—due to math operations—every other number in the model also eventually becomes a NaN.
  - Helps the model learn appropriate weights for each feature. Without feature scaling, the model will pay too much attention to the features having a wider range.

* Handling extreme outliers
  - log(series values)
  - set boundary (mean+3sd or upper/lower)
  - binning. Instead of having one floating-point feature, divide feature into "bins", which N distinct boolean features.

* Omitted values: missing values
* Duplicate examples.
* Bad labels: wrong label.
* Bad feature values: wrong values.

### Feature Crosses
* Encoding Nonlinearity
  - nonlinear problem: to split nonlinear dots
  - features are continuously feature
  - feature cross is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together
  - use feature cross like other features in linear formula
    + [A X B]: a feature cross formed by multiplying the values of two features.
    + [A x B x C x D x E]: a feature cross formed by multiplying the values of five features.
    + [A x A]: a feature cross formed by squaring a single feature.
* Crossing One-Hot Vectors
  - feature crosses of one-hot feature vectors as logical conjunctions
  - features here are categorical feature
  - A(NY, CA) B(English,Spanish) --> feature crossL: NY and English, NY and Spanish, CA and English, CA and Spanish




## ------- machine learning -------
### supervised algorithm methods

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



### clustering algorithms
* K-Means [wiki](https://en.wikipedia.org/wiki/K-means_clustering)
  - unsupervised algorithm which  solves the clustering problem
  - Given a set of observations (x1, x2, …, xn), where each observation is a d-dimensional real vector, k-means clustering aims to partition the n observations into k (≤ n) sets S = {S1, S2, …, Sk} so as to minimize the within-cluster sum of squares (WCSS). ![wcss](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-ml_term-kmeans-wcss.png)
  - Because the total variance is constant, this is also equivalent to maximizing the squared deviations between points in different clusters (between-cluster sum of squares, BCSS).![bcss](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-ml_term-kmeans-bcss.png)
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
 


### Ensemble learning
* wiki link: https://en.wikipedia.org/wiki/Ensemble_learning

### representation learning

* autoencoder



### validation
* binary classification: confusion table/confusion matrix

||..||Actual True||Actual False||
|Predict postive|True Positive (TP)|False Positive (FP)|
|Predict negative|False Negative (FN)|True Negative (TN)|
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