## ------- clustering algorithms -------
* type of clusters
  - partitional: create only a single set of clusters
  - hierarchical: create separate sets of nested clusters, each cluster in its hierarchal level
* cluster similarity metrics
  - Euclidean metric: a measure of distance between data points on a Euclidean plane
  - Manhattan metric: a measure of distance between data points where distance is sum of absolute value of differences between 2 points Cartesian coordinates
  - Minkowski distance metric: a generalization of Euclidean and Manhattan metrics
  - Cosine similarity metric: based on orientation (cosine of the angle between 2 points)
  - Jaccard distance metric: for non-numeric data(text), generate numeric index value between text strings


* sample data comparison with python[link](http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py)![cluster_comparison](https://github.com/karina7rang/notes/blob/master/machine_learning/picture/machine_learning-clustering.png)

### distance
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

* Agglomerative Hierarchical Clustering
  - Hierarchical clustering algorithms actually fall into 2 categories: top-down or bottom-up.
  - Bottom-up hierarchical clustering is therefore called hierarchical agglomerative clustering or HAC.
  - each data point as a single cluster, select a distance(average linkage) metric that measures the distance between two clusters, combined 2 clusters with the smallest average linkage.

* Dimensionality Reduction Algorithms
  - identify highly significant variable(s)
  - dimensionality reduction algorithm helps us along with various other algorithms like Decision Tree, Random Forest, PCA, Factor Analysis, Identify based on correlation matrix, missing value ratio and others.


### density
* Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
  - a density based clustered algorithm similar to mean-shift
  - with e distance moving center, data points are in one cluster. iteration till all data processed.
  - advantage: num center is unknown, outliers as noises and ignored
  - disadvantage: 
    + not good for clusters are of varying density
    + e is unknown
    + computationally expensive
    + cluster size and density are empirical

* KDE (kernel density estimation)
  - place a kernel on each data point and then sum the kernels to generate a kernel density
  - kernel: a weighting function that is useful for quantifying density

### distribution
* Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)
  - GMMs assume the data points are Gaussian distributed
  - two parameters to describe the shape of the clusters: the mean and the standard deviation
  - use an optimization algorithm called Expectation–Maximization (EM) to find mean and sd
  - compute the probability that each data point belongs to a particular cluster. maximize the probabilities of data points within the clusters
  - K-Means is actually a special case of GMM in which each cluster’s covariance along all dimensions approaches 0.
  - advantage: good for any ellipse shape, Secondly, since GMMs use probabilities, they can have multiple clusters per data point.

