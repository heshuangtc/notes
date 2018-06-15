## feature engineering
* good features [link](https://developers.google.com/machine-learning/crash-course/representation/qualities-of-good-features)
  - Avoid rarely used discrete feature values
  - Prefer clear and obvious meanings
  - Don't mix "magic" values with actual data. To work around magic values, convert the feature into two features:One feature holds only quality ratings, never magic values. One feature holds a boolean value indicating whether or not.
  - The definition of a feature shouldn't change over time

## clean data [link](https://developers.google.com/machine-learning/crash-course/representation/cleaning-data)
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

## Feature Crosses
### Encoding Nonlinearity
* nonlinear problem: to split nonlinear dots
* features are continuously feature
* feature cross is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together
* use feature cross like other features in linear formula
  - [A X B]: a feature cross formed by multiplying the values of two features.
  - [A x B x C x D x E]: a feature cross formed by multiplying the values of five features.
  - [A x A]: a feature cross formed by squaring a single feature.
### Crossing One-Hot Vectors
  - feature crosses of one-hot feature vectors as logical conjunctions
  - features here are categorical feature
  - A(NY, CA) B(English,Spanish) --> feature crossL: NY and English, NY and Spanish, CA and English, CA and Spanish