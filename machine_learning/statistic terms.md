### statistical terms
*  uniform distribution: A distribution in which all values have the same frequency.
*  effect size: a summary statistic intended to describe the size of an
effect. For example, to describe the difference between two groups, one obvious choice
is the difference in the means.
*  probability mass function (PMF): a representation of a distribution as a function that maps from values to probabilities.
* cumulative distribution function (CDF): The CDF is a function of x, where x is any value that might appear in the distribution. To evaluate CDF(x) for a particular value of x, we compute the fraction of values in the distribution less than or equal to x.
* percentile rank: The percentage of values in a distribution that are less than or equal to a given value.
* percentile: The value associated with a given percentile rank.
* median: The 50th percentile, often used as a measure of central tendency.
* interquartile range: The difference between the 75th and 25th percentiles, used as a measure of spread.
* quantile: A sequence of values that correspond to equally spaced percentile ranks; for example,
the quartiles of a distribution are the 25th, 50th and 75th percentiles.
* kurtosis: flat or sharp
* skewness: tail longer one right or left
* covariance: relationship 2 variables (negative/positive)
* correlation: linear relationship between 2 variables
* control limit [link](https://www.spcforexcel.com/knowledge/variable-control-charts/xbar-r-charts-part-1)
  - range: max - min
  - LCL: mean-3sd
  - UCL: mean + 3sd
* 
* `scipy.stats.norm` functions [link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)

![cdf_pdf_pmf_relationship](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/statistic_term-cdf_pdf_pmf_relationship.PNG)

* z test
  - z score: x-mean()/standard deviation()
  - [wiki link](https://en.wikipedia.org/wiki/Z-test)
  - standard error(SE): `sd/sqrt(n)`or`square root((p-p^2)/n)`
  - 2 sample testing [link](https://www.cliffsnotes.com/study-guides/statistics/univariate-inferential-tests/two-sample-z-test-for-comparing-two-means)
  - confidence interval `mean+-z_score*SE`
* t test
  - when unknown population/sample standard deviation, use number of sampling to calculate t test statistic
  - 

* defining the different kinds of variables
  - A categorical variable, also called a nominal variable, is for mutual exclusive, but not ordered, categories. 
  - A ordinal variable, is one where the order matters but not the difference between values.
  - A interval variable is a measurement where the difference between two values is meaningful.
  - A ratio variable, has all the properties of an interval variable, and also has a clear definition of 0.0. When the variable equals 0.0, there is none of that variable. 

#### A/B Testing
* make a change, see if it significantly improves a goal

* binomial distribution
  - with parameters N and p is the discrete probability distribution of the number of successes in a sequence of N independent experiments, each asking a yes–no question [wiki](https://en.wikipedia.org/wiki/Binomial_distribution)
  - determine a binomial distribution: 2 types of outcomes(T/F) + independent events + identical distribution(p same for all)
  - some terms - 1 distribution
    + probability(p) = #yes/#cases = #yes/N
    + std deviation(SE) = square_root(p(1-p)/N)
    + margin of error(m) = Z_score * SE = Z_score * square_root(p(1-p)/N)
    + error range/confidence interval = p+-m
  - hypothesis testing
    + null/H0: p_countrol = p_experiment
    + alternative/Ha: p_countrol != p_experiment
  - some terms - compare 2 distributions
    + p_pool = (#yes_control+#yes_experiment)/(N_control+N_experiment)
    + SE_pool = square_root( p_pool*(1-p_pool)*(1/N_control+1/N_experiment) )
    + estimated difference/error(d) = p_experiment-p_countrol = #yes_control/N_control - #yes_experiment/N_experiment (H0:d=0, d is binomial distribution. with 9x% confidence, if d out of confidence interval/error range,reject)
    + alpha: p(reject null|null true)
    + beta: p(fail to reject|null false)
    + sensitivity(1-beta): usually 80%
    + error range/confidence interval = d+-m

* measurements
  - click-through-rate : #click(1 user can click many times)/#visits
  - click-through-probability: #if click(#unique user click)/#users(#visits)

* calculating variability
  - probability: distribution(binomial)/estimated variance(p(1-p)/N)
  - mean: distribution(normal)/estimated variance(sd2/N)
  - median/percentile: distribution(depends on data)/estimated variance(depends on data)
  - count/difference: distribution(normal often)/estimated variance(var(x)+var(y))
  - rates: distribution(Poisson)/estimated variance(mean(x))
  - ratios: distribution(depends on data)/estimated variance(depends on data)

*

*
