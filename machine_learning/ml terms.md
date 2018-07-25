## basic 
* types of analytics
  - descriptive analytics: what happened?
  - diagnostic analytics: why this happened?
  - predictive analytics: what will happen?
  - prescriptive analytics: what can improve?
* types of algorithms
  - supervised
  - unsupervised
  - reinforcement : semi-supervised. improve model behavior based on rewards
![ml algorithm map](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-ml_algomap.png)
* types of algorithms
  - association rule learning algorithm: discover associations between features
  - instance-based algorithm: classify based on similarity
  - regularizing algorithm: prevent model overgitting or solve an ill-posed problem
  - naive bayes method: predict likelihood of an event occurring
  - decision tree:
  - clustering algorithm
  - dimension reduction method: remove redundant information, noise and outliers, dimension reduction analysis
  - neural network: use layers of interconnected units
  - deep learning method:incorporate transitional neural networks
  - ensemble algorithm: 
>> reference: Data Science for Dummies, Chapter 4


## ------- data preparation -------
### dimension reduction
* PCA (principle component analysis)
  - for uncorrelated features, present most information
  - output: reduced set of meaningful, non-information-redundant principle variables
  - assumptions:
    + multivariate normal features
    + continuous features
  - how many components should use
    + top n% components
    + trails on train/test and based on performance
* SVD(singular value decomposition)
  - reduce number of columns
  - `A = u * S * v` 
  - A(original matrix) `m*n`
  - u(left orthogonal matrix: holds important non-redundant information about observations) `m*r`
  - v(right orthogonal matrix: holds important non-redundant information about columns) `r*n`
  - S(diagonal matrix: information about the decomposition processes performed during the compression) (square rot of eigenvalue of A) `r*r`
* factor analysis
  - filter out redundant information and noise. the greater feature variance, the more information feature contains. find cause of shared variance.
  - output: reduced set of meaningful, non-information-redundant latent variables
  - assumptions:
    + features are numeric variables (continuous or ordinal)
    + 100+rows and 5+columns
    + Pearson's R > 0.3 between features



* correlation
  - Pearson's R: linear relationship (1 positive, 0 none, -1 negative) for numeric continuous variable, normal distribution
  - Spearman's rank correlation: detect correlation between ordinal variable, non-linearly, non-normally distribution
  - 



## ------- validation -------

### logistic regression
* confusion matrix(for binary classification)
  - confusion table/confusion matrix

|..|Actual True|Actual False|
|-------|-------|-------|
|Predict postive|True Positive (TP)|False Positive (FP) type I error|
|Predict negative|False Negative (FN) type II error|True Negative (TN)|

  - accuracy = `#correct predictions/total # predictions` = `(TP+TN)/(TP+TN+FP+FN)`
  - precision = `TP/(TP+FP)` What proportion of positive identifications was actually correct? more important when care predict is real actual and not include more.
  - recall = `TP/(TP+FN)` What proportion of actual positives was identified correctly? more important when care predict include more actual.

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

* feature scaling will affect svm(rbf kernel) and kmeans instead of decision tree or linear regression



## ------- other -------
alphago zero nature article [link](https://www.nature.com/nature/journal/v550/n7676/full/nature24270.html)

field-aware factorization machines

factorization machines

### representation learning

* autoencoder