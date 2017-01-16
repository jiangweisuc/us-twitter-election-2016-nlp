import glob, sys
from six.moves import cPickle as pickle
from TwitterElection.PreprocessTweets import PreprocessTweets

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
    fnames1 = glob.glob(INPUT_DIR + '*')
    fnames = sorted(fnames1)

    PreprocessTweets().test()

    tweets, label = PreprocessTweets().preprocess(fnames)
    save_pickle(OUTPUT_DIR + 'TwitterElection2016_tweets_en', tweets)
    save_pickle(OUTPUT_DIR + 'TwitterElection2016_label_en', label)

    print(len(tweets))
    print(tweets[0])
    print('Size of tweets in memory: ', sys.getsizeof(tweets))


if __name__ == "__main__":
    main()
