## Natural Language Processing

* intro with python [link](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
* 


### data preparing
* remove noise words: such as 'am','of','is' etc
* lexicon normalization: multiple representations exhibited by single word, such as 'play', 'player', 'played', 'plays' and 'playing' are the different variations of the word – 'play'
  - Stemming: a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word
  - Lemmatization: an organized & step by step procedure of obtaining the root form of the word, it makes use of vocabulary (dictionary importance of words) and morphological analysis (word structure and grammar relations).
* remove/replace words or phrases not in lexical dictionaries, such as acronyms, hashtags with attached words, and colloquial slangs


### feature engineering
* Dependency Trees features: The relationship among the words in a sentence is determined by the basic dependency grammar.
* Part of speech tagging features
  - every word in a sentence is associated with a part of speech (pos) tag (nouns, verbs, adjectives, adverbs etc).
  - disambiguation: same word has 2 different context
* Entity extraction features
  - Named Entity Recognition (NER): person names, location names, company names etc
  - topic: Latent Dirichlet Allocation (LDA)
  - N-Grams: A combination of N words together.
* Statistical Features
  - Term Frequency – Inverse Document Frequency (TF – IDF): gives the relative importance of a term in a corpus (list of documents)
  - Count / Density / Readability Features
* Word Embedding
  - redefine the high dimensional word features into low dimensional feature vectors
  - widely used in deep learning models such as CNN or RNN
* text classification
  - use cases: Email Spam Identification, topic classification of news, sentiment classification and organization of web pages by search engines.
* text matching/similarity
  - Phonetic Matching: takes a keyword as input and produces a character string that identifies a set of words that are (roughly) phonetically similar.
  - Cosine Similarity: output is percentage sum(str1*str2) / sqrt(sum(str1^2))*sqrt(sum(str2^2))