# Twitter Presidential Election 2016 USA Project

Melissa K (Jan 2017)

##### Table of content

- [Motivation](#motivation)
- [Data Collection](#data-collection)
- [Preprocessing Steps Tweets](#preprocessing-steps-tweets)
- [Word2vec Embedding](#word2vec-embedding)
- [Classification](#classification)
- [Deep Learning](#deep-learning)
- [What I learned](#what-i-learned)


Motivation
---

The U.S. Presidential Election on Nov 8, 2016 (Donald Trump vs. Hillary Clinton) has
been a historical event, where the outcome turned out
to be contrary to most expert predictions. The collected Twitter Streaming Data has the potential to uncover a shift in opinions and reactions
around the Election Day.

Data Collection
---
Prior to the Election Day, I wrote a pretty standard Python module based on
the [Tweepy](http://docs.tweepy.org/en/v3.5.0/) library.
I started data collection on Nov 7 around 3 pm and stopped the
live stream on Nov 9 around 1pm. I collected ~40GB of raw Twitter
Streaming Data. This was probably just a very tiny fraction of all tweets. Each tweet was saved in json format.
The [Twitter Streaming API](https://dev.twitter.com/streaming/overview) had
the advantage of collecting a much higher volume of data than querying
the [Twitter REST API](https://dev.twitter.com/rest/public) using a regular free account. I collected
tweets in English, German, French and Spanish ```['en', 'de', 'fr', 'es']```
and tracked the following tags ```['#Trump', '@realDonaldTrump', '@Always_Trump',
                        '#MakeAmericaGreatAgain', '@HillaryClinton',
                        '@VoteHillary2016', '#HillaryClinton', '@ClintonNews']```.
As a next step I parsed the raw tweets and saved the ```['lang'], ['text']``` and ```['created_at']``` values
in tabular format (simple delimiter separated file with header) in different sub-folders according to the
language tag.
This way the data volume was reduced to around 1.5GB. Subsequent data analysis focused on English,
however I plan to extend my NLP capabilities to more languages in the future.


Preprocessing Steps Tweets
---

A tweet text has max. 140 characters and there are a multitude of decisions to be made in terms of preprocessing
before each tweet is formalized enough for mathematical modeling.
I came up with the following steps (see [PreprocessTweets.py](TwitterElection/PreprocessTweets.py)):

1. Removing several characters, such as @-mentions, #-mentions, 'RT', everything following an apostrophe,
 some custom characters such as numbers/special characters (e.g.: ?!.,&<') or URLs via regex matching.
2. Emojis were left as they are (for fun there is a function that extracts emojis).
3. Tokenizing words using nltk's [TweetTokenizer](http://www.nltk.org/api/nltk.tokenize.html). Note
 smiley strings like :-)))))))))), :-( or :P are tokenized to one token while repeated characters are
 reduced to one. Neat feature of TweetTokenizer.
4. Removing stopwords and only now removing single : or - or ) etc. characters
5. Spell corrections (simple naiv approach: using first suggestion of [PyEnchant](http://pythonhosted.org/pyenchant/))
6. Lastly stemming words using nltk's [SnowballStemmer](http://www.nltk.org/_modules/nltk/stem/snowball.html).


More details on how I handled above steps can be found in the source code (see [PreprocessTweets.py](TwitterElection/PreprocessTweets.py)).
Below is a random test tweet I fabricated to debug the code. You can see the results of the preprocessing
procedures. There are certainly pros and cons for my approach and
I really believe it's an artwork at the end. Also some cases are not fully accounted for, e.g. '🇺🇸' emoji got split
into separate tokens and 'Makesme' did not get separated into two words...not to mention some funny
spell corrections. However, these steps may have helped to more uniformly derive one token for a multitude of
variations of the "same word".

```python
Original tweet:  RT #Election2016 🇺🇸 @realDonaldTrump :-)))) victory :( :P #HillaryClinton 😡 or 😄 @NYSE that's 4% x &gt; 2 &amp I luv this, Makesme sooooo haaaaaapppy :) ALL OF US https://www.twitter.com/
Tokenized and preprocessed tweet:  ['🇺', '🇸', ':-)', 'victori', ':(', ':p', '😡', '😄', 'x', 'lug', 'makes m', 'sou', 'happi', ':)', 'us']
Emojis extracted:  ['🇺🇸', '😡', '😄']
```

Word2vec Embedding
---


Word2vec is truly booming, not only in Natural Language Processing. It is an embedding approach with the goal of deriving a "continuous vector
space where semantically similar words are mapped to nearby points
 ('are embedded nearby each other')" instead of "representing words as unique/discrete IDs".
 [Tensorflow word2vec](https://www.tensorflow.org/tutorials/word2vec/) explains the background extremely well, highly recommend
 reading their documentation as I am not going into further details here!
 As opposed to Tensorflow, I wanted to explore [gensim word2vec](https://radimrehurek.com/gensim/models/word2vec.html) here in this project
(see [Embedding.py](TwitterElection/Embedding.py)).

Following parameters were set:

- Method: Continuous Bag-of-Words model (CBOW)
- Embedding Dimension 200 (in ```gensim``` size=200)
- No maximum vocabulary size (note this is limit RAM and not max words)
- No maximum word length as opposed to different approaches. Not needed here as I took averages of the
embedding vectors corresponding to each token of one tweet, therefore the number of tokens did not matter (see below). Other
approaches define a maximum sentence length and flatten the embedding vectors, which however results in an
extremely high feature space.
- Remaining parameters pretty much default

I believe that when it gets to embeddings it is most important to track the dimensions of the data/matrices in
the process. Here there are 5370247 tweets, in ```gensim``` language
this would be 5370247 sentences. The tweets are stored as a Python list of tweets, whereas each tweet
is a list of tokens (therefore list of lists). I didn't set a RAM limit to the vocabulary size and
ended up with 14202 words that make up the vocabulary. Therefore, the resulting model matrix of the word2vec has dimensions (14202, 200).

```python
Total number of tweets: 5370247
Size of tweets in memory:  43281112
Length of Word2Vec Vocabulary:  14202
Dimension of embedding (max_vocab_size x dim): (14202, 200)
```

The next step is to transform the same 5370247 tweets using this model (see [Embedding.py](TwitterElection/Embedding.py)). For
each tweet each token will be compared with the model vocabulary and if the token is contained within the
vocabulary the respective vector will be extracted, otherwise a vector of zeros with the same size 200 will be "extracted".
Then as a simple approach the average of all of the tokens of one tweet was taken. The final feature matrix X
thus has dimensions tweets x dim (5370247, 200). Below I also show some neat features of the word embedding, such as quantifying
the similarity of two words or finding vocabulary that is 'close' to a specific word.


```python
Dimension of X: (5370247, 200)
Most similar to trump [('clinton', 0.4256572127342224), ('hillari', 0.3691369891166687), ('would', 0.3448511064052582), ('thu', 0.33279508352279663), ('say', 0.32681870460510254), ('popular', 0.3263714611530304), ('candid', 0.32262635231018066), ('drum pf', 0.3211328685283661), ('peopl', 0.3207015097141266), ('whether', 0.31549662351608276)]
Most similar to poepl [('pol', 0.7333731651306152), ('everyon', 0.4832013249397278), ('voter', 0.4737037420272827), ('anyon', 0.4535198211669922), ('support', 0.4492112994194031), ('guy', 0.4350971579551697), ('folk', 0.4311058819293976), ('realli', 0.4233342409133911), ('person', 0.40902650356292725), ('women', 0.3935454785823822)]
Similarity Hillary and Trump 0.369136993042
```

Classification
---

Instead of performing a classic sentiment analysis, I labeled each tweet as hillary (encoded as int 0), trump (encoded as int 1) or neutral (encoded as int 2)
based on whether # and @ mentions of the respective tweet were dominated by tags pointing to either Trump or Hillary. If none was the case the neutral label got assigned (see [PreprocessTweets.py](TwitterElection/PreprocessTweets.py)).
This approach resembles a topic classification. Note that all # and @ mentions were removed after labeling the respective tweet in the course of
preprocessing the tweet, so they were not words used as features.

For the classification task, solely tweets labeled as hillary or clinton were used. Note that tweets about trump were much (!) more frequent.
To have a balanced and manageable data set, I selected only a subset of labeled tweets with an even amount of samples for each of the two labels.
[TopicModel.py](TwitterElection/TopicModel.py) contains a class that wraps around the complete machine learning classification task. Main steps in bullet points:

Input data (feature matrix X and target vector) for topic classification:

- The feature matrix X (numpy nd-array: samples x features, here (157982, 200)) was derived using the embedding model ([Word2vec Embedding](#word2vec-embedding)). 200 was the chosen embedding dimension.
- The target vector containing the label for each sample is a numpy 1d-vector of shape (samples,), here (157982,)

Machine Learning procedures (train, optimize and predict):

- 3-fold cross-validation was done using sklearn's [model_selection.StratifiedShuffleSplit](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html).
- Selected best model from the following three models: [NaiveBayesClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html), [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [AdaBoostClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html).
- As first step parameters for each of these three models were optimized using the less computationally expensive [model_selection.RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) from the sklearn library.
To further optimize this step a custom scorer [metrics.make_scorer](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) that maximizes the minimum of precision and recall was used instead of accuracy as performance metric.
- When looping over the three classifiers (clfs) using their *best* parameters, metrics were stored to finally derive a Confidence Interval (CI) for each of the classifiers. If lower and upper ends of two classifiers don't overlap the models are said to be significantly different from one another.
- Accuracy on the validation set was used to determine final model performance.

Output of [main_TwitterElection_classification.py](main_TwitterElection_classification.py):

```
Total number of tweets: 5370247
Total number of filtered and labeled tweets: 157982
Shapes of X and target:  (157982, 200) (157982,)
Parameter Search...
Parameter Search of... RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=1, oob_score=False, random_state=42,
            verbose=0, warm_start=False)
Parameter Search of... AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=None)
Best Estimators:  [GaussianNB(priors=None), RandomForestClassifier(bootstrap=False, class_weight=None,
            criterion='entropy', max_depth=1, max_features=2,
            max_leaf_nodes=None, min_impurity_split=1e-07,
            min_samples_leaf=9, min_samples_split=6,
            min_weight_fraction_leaf=0.0, n_estimators=31, n_jobs=1,
            oob_score=False, random_state=42, verbose=0, warm_start=False), AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.8, n_estimators=100, random_state=None)]

Cross-Validation using following Classifier:

GaussianNB(priors=None)


accuracy     0.671588
precision    0.682339
recall       0.671588
f1_score     0.666677
dtype: float64
Confidence Interval (CI) Accuracy: (0.66781321431938101, 0.67536298135630035)

Cross-Validation using following Classifier:

RandomForestClassifier(bootstrap=False, class_weight=None,
            criterion='entropy', max_depth=1, max_features=2,
            max_leaf_nodes=None, min_impurity_split=1e-07,
            min_samples_leaf=9, min_samples_split=6,
            min_weight_fraction_leaf=0.0, n_estimators=31, n_jobs=1,
            oob_score=False, random_state=42, verbose=0, warm_start=False)


accuracy     0.647788
precision    0.656927
recall       0.647788
f1_score     0.642584
dtype: float64
Confidence Interval (CI) Accuracy: (0.64072072917132639, 0.65485525762045982)

Cross-Validation using following Classifier:

AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.8, n_estimators=100, random_state=None)


accuracy     0.731352
precision    0.732381
recall       0.731352
f1_score     0.731054
dtype: float64
Confidence Interval (CI) Accuracy: (0.72825247897875744, 0.73445172349092558)
```

### Visualizations

All Classifiers ROC curves

<img src="output/all_clfs_roc_curve.png" width="400">

All Classifiers Confusion Matrix

<img src="output/all_clfs_confusion_matrix.png" width="800">

### Best Classifier AdaBoost Confusion Matrix in large:

<img src="output/AdaBoostClassifier_confusion_matrix.png" width="400">


### Conclusion Machine Learning Topic Classification

- AdaBoost showed best accuracy on validation data set.
- Preprocessing and word2vec embedding dimensions are parameters that have been left out to further fine-tune.
- About 73% accuracy doesn't sound too impressive, but it appears that humans as well only agree about 70% of the time on topics auch as sentiment analysis, which is similar to the topic modeling approach used here (see [wikipedia sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)).
- Only a subset of all data could be processed in a reasonable time on the local computer.
- All metrics (aka precision, recall and accuracy) were similar, pointing out that there were no misleading high false positives and false negatives hidden by a higher accuracy score.
- It appears that the word embedding aka features were better predictors of tweets about trump. One reason for this could have been that the original data set (all tweets) was used to train the word2vec embedding model and that data set was somewhat unbalanced in that in general more tweets were referencing trump.


Deep Learning
---

Intend to experiment with Recurrent Neural Network (RNN) using the Long Short Term Memory (LSTM) architecture.


What I learned
--------------

- Preprocessing tweets is an artwork
- Explored gensim word2vec embedding
- Tested Twitter topic modeling in the context of U.S. Presidential Election
- Explored Deep Learning - LSTM
