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


## ------- time series algorithm -------
