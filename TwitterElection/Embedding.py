from gensim.models.word2vec import Word2Vec
import numpy as np


# Python 3.5 environment
# Last updated: Jan 5, 2017
# Written by Melissa K

class MeanEmbeddingVectorizer(object):

    def __init__(self):
        pass

    # gensim docs
    # https://radimrehurek.com/gensim/models/word2vec.html

    @staticmethod
    def fit_word2vec(tweets, dim=200, max_vocab_size=None):
        """
        Fit word2vec to list of preprocessed tweets, in official gensim docs input is called "sentences"
        model.syn0 has dimension (max_vocab_size x dim)
        """
        model = Word2Vec(tweets, size=dim, max_vocab_size=max_vocab_size, window=5, min_count=45, workers=4)
        model.save('model/Election2016_W2V_gensim_model_en')
        print('Length of Word2Vec Vocabulary: ', len(model.vocab))
        print('Dimension of embedding (max_vocab_size x dim):', model.syn0.shape)
        return model

    @staticmethod
    def transform(tweets, dim=200):
        dim = dim
        model = Word2Vec.load('model/Election2016_W2V_gensim_model_en')
        return np.array([np.mean([model[w] for w in tweet if w in model.vocab]
                                 or [np.zeros(dim)], axis=0) for tweet in tweets])

