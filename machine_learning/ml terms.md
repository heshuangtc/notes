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
