import numpy as np
import random
from six.moves import cPickle as pickle
from TwitterElection.Embedding import MeanEmbeddingVectorizer
from TwitterElection.TopicModel import TopicModel
from gensim.models.word2vec import Word2Vec

# Python 3.5 environment
# Last updated: Feb 19, 2017
# Written by Melissa K

INPUT_DIR = '/path/to/data/en/' # subfolder with all english tweets only
OUTPUT_DIR = '/path/to/data/'


def save_pickle(fname, tweets):
    with open(fname + '.pickle', 'wb') as f:
        pickle.dump(tweets, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(fname):
    with open(fname + '.pickle', 'rb') as f:
        tweets = pickle.load(f)
    return tweets


def main():
    global INPUT_DIR, OUTPUT_DIR

    # data loaded are preprocessed tweets and labels done in PreprocessTweets.py
    tweets_all = load_pickle(OUTPUT_DIR + 'TwitterElection2016_tweets_en')
    print('Total number of tweets:', len(tweets_all))
    # Here Embedding already done in a previous main code, model was saved to disk
    # MeanEmbeddingVectorizer.fit_word2vec(tweets, dim=200, max_vocab_size=None)

    # get labels according to weather tags for hillary (label 0) or trump (label 1) were larger
    label = load_pickle(OUTPUT_DIR + 'TwitterElection2016_label_en')
    label_0 = [i for i, x in enumerate(label) if x == 0]  # hillary (much less)
    label_1 = [i for i, x in enumerate(label) if x == 1]  # trump
    new_len = int(len(label_0) / 8)
    label_0 = [label_0[i] for i in sorted(random.sample(range(len(label_0)), new_len))]
    label_1 = [label_1[i] for i in sorted(random.sample(range(len(label_1)), new_len))]
    label_indices = label_0 + label_1

    target = np.array([label[i] for i in label_indices])
    tweets = [tweets_all[i] for i in label_indices]
    print('Total number of filtered and labeled tweets:', len(tweets))

    # Do embedding using the model saved to disk from step MeanEmbeddingVectorizer.fit_word2vec(tweets, dim=200, max_vocab_size=None)
    X = MeanEmbeddingVectorizer().transform(tweets)

    print('Shapes of X and target: ', X.shape, target.shape)

    TopicModel(X, target).cv_model_eval(param_search=1, load_best_model=None)  # use this for complete train, optimize and predict and plotting
    #TopicModel(X, target).cv_model_eval(param_search=None, load_best_model=1)  # use this when model already trained and only plotting is desired


if __name__ == "__main__":
    main()
