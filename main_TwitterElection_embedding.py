import glob, sys
from six.moves import cPickle as pickle
from TwitterElection.Embedding import MeanEmbeddingVectorizer
from gensim.models.word2vec import Word2Vec

# Python 3.5 environment
# Last updated: Jan 5, 2017
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

    tweets = load_pickle(OUTPUT_DIR + 'TwitterElection2016_tweets_en')

    print('Total number of tweets:', len(tweets))
    print(tweets[0])
    print('Size of tweets in memory: ', sys.getsizeof(tweets))

    MeanEmbeddingVectorizer.fit_word2vec(tweets, dim=200, max_vocab_size=None)
    # note max_vocab_size is limit RAM not number of words!

    X = MeanEmbeddingVectorizer().transform(tweets)
    print(X.shape)

    # apart from above let's try some fun stuff with the model
    model = Word2Vec.load('model/Election2016_W2V_gensim_model_en')
    result = model.most_similar(positive=['trump'])
    print('Most similar to trump', result)

    result = model.most_similar(positive=['peopl'])
    print('Most similar to poepl', result)

    result = model.similarity('hillari', 'trump')
    print('Similarity Hillary and Trump', result)


if __name__ == "__main__":
    main()
