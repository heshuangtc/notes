## basic knowledge
* recommendation types
  - product recommendation: amazon, ebay, bank
  - movie recommendation: netflix
  - music recommendation: spotify
  - connection recommendation: facebook

## recommender algorithm
* popularity-based recommender
  - based on popularity to recommend group of people
  - advantage: dont have to know product/item
  - disadvantage: products have to have reviews/rating. hard to new users. popular will have more reviews/raings
  - statistic measurement
    + rating counting (recommend most popular)
    + rating mean (most like)
  - correlation of statistic measurement to make recommendation
    + recommend item that is most similar to an item users (here are all users) have already chosen
    + Pearson correlation coefficient (r) which is linear relationship

* classification-based collaborative filtering recommender
  - based on other users rating history
  - personalization data:
    + user/item attribute data
    + purchase history data
    + contextual data
  - new user info is predict input, history users behaviors are train input and label
  - classification algorithms
    + Bayes classification
    + logistic regression (binary)

* correlation collaborative filtering recommender - SVD (singular value decomposition)
  - utility matrix is sparse matrix (user-row, item-col, rating-values)
  - SVD reduct # cols, if recommend item, then item needs to be row when apply SVD
  - a linear algebra method that can decompose a utility matrix into three compressed matrices
  - `A = u * S * v` 
    + A(original matrix-utility matrix) 
    + u(left orthogonal matrix: holds important non-redundant information about users) v(right orthogonal matrix: holds important non-redundant information about items) 
    + S(diagonal matrix: information about the decomposition processes performed during the compression)
  - correlation coefficient on SVD output matrix(item-row,compressed user-col)
  - recommendation based on correlation

* score collaborative filtering recommender
  - score = user review rating by movie category * movie rating by category
  - workflow
    + random give score matrix
    + cost function contain difference between real score and random score
    + apply optimization algorithm to get minimum cost
    + then score with min cost will be estimated score, which can be used for recommendation
  - adjust movie rating by category
    + new user adjust movie rating = avg(movie rating all users) + 0
    + old user adjust movie rating = avg(movie rating all users) + (avg(old user rating)-avg(all user rating))

* content-based recommender
  - based on item features/attributes similarity with other items
  - algorithm: nearest neighborer (unsupervised method) to find closest recommendation
  - advantage: product without user reivews
  - disadvantage: every product needs attributes/features