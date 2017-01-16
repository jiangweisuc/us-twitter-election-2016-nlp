import re
import html
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import enchant
from enchant.checker import SpellChecker

stemmer = SnowballStemmer("english")

# Python 3.5 environment
# Last updated: Jan 5, 2017
# Written by Melissa K

class PreprocessTweets(object):
    name = 'PreprocessTweets'
    """
    Preprocesses raw Twitter tweets
    Main steps: cleaning (such as removing unwanted or repeated characters/symbols, stop words and spell checking), stemming and tokenizing
    The preprocessed tweets are in a format ready to perform Machine Learning
    There are probably many ways to improve on it, but it covers the basics...
    """

    def __init__(self):
        self.l_trump = ['#Trump', '@realDonaldTrump', '@Always_Trump', '#MakeAmericaGreatAgain']
        self.l_hillary = ['@HillaryClinton','@VoteHillary2016', '#HillaryClinton', '@ClintonNews']
        self.trump_pattern = re.compile('%s' % '|'.join(self.l_trump))
        self.hillary_pattern = re.compile('%s' % '|'.join(self.l_hillary))
        self.l_sub_pattern = ['[.,&><%\d+!\?…]',  # several custom
                              '(?<=\')\w+',  # everything following an apostrophe
                              '\'’',
                              'RT', # RT stands for retweeted
                              '(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', # hashtags
                              '(?:@[\w_]+)',  # @-mentions
                              'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
                              ]
        self.sub_pattern = re.compile('%s' % '|'.join(self.l_sub_pattern))
        self.emoji_pattern = re.compile("[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6C0" "\U0001F170-\U0001F251" "]+")
        self.d = enchant.Dict("en_US")
        self.chkr = SpellChecker("en_US")
        self.l_stopwords_en = stopwords.words('english') + ['e', ':', ')', '(', 'u', '-', '"', '\'']

    def tokenize_tweet(self, tweet):
        """
        Takes raw tweet string
        Removes several characters, such as @-mentions or URLs via regex matching
        Removes stopwords, corrects spelling (simple naiv approach)
        Stems lower case strings
        :param tweet: raw tweet string
        :return: tokens (list of preprocessed words)
        """
        tweet = self.sub_pattern.sub(' ', html.unescape(tweet))
        tokens = TweetTokenizer(reduce_len=True).tokenize(tweet)
        tokens = [stemmer.stem(self.correct_spelling(t)) for t in tokens if t.lower() not in self.l_stopwords_en]
        return tokens

    def correct_spelling(self, t):
        """
        Using enchant library to check spelling of word
        :param t: word
        :return: corrected word if applicable (spell check failed)
        """
        if self.d.check(t) is False and re.search('[@#:)(]', t) is None:
            try:
                t1 = self.d.suggest(t)[0]
                if t1 not in ['e']: # just something I noticed that is particular to enchant suggestions
                    t = t1
            except IndexError:
                pass
        return t.lower()

    def get_len_trump(self, tweet):
        """
        Get number of mentions of the Trump related # (See list above)
        """
        return len(self.trump_pattern.findall(tweet))

    def get_len_hillary(self, tweet):
        """
        Get number of mentions of the Hillary related # (See list above)
        """
        return len(self.hillary_pattern.findall(tweet))

    def get_emoji(self, tweet):
        """
        Return list of emojis from tweet (regex matches can be extended on...)
        """
        return self.emoji_pattern.findall(tweet)

    def preprocess(self, fnames):
        """
        This function should be called to perform preprossing of all tweets in all files
        :param fnames: list of all file names (each new line contains lang|text|created_at (pipe separated text file))
        :return: tweets (list of preprocessed tweets, label (hillary [0] vs trump [1] vs unknown [2]))
        """
        tweets = []
        label = []
        for i, fname in enumerate(fnames):
            print(i, fname)

            with open(fname, 'r') as f:
                for _ in range(0, 1): # skip header
                    next(f)
                for line in f:
                    try:
                        line = line.split('|')
                        if len(line)>2:
                            # assign a label to the raw tweet based on whether # and @ mentions were dominated
                            # by either Trump or Hillary or neither (note @ mentions will be removed later)
                            len_trump = self.get_len_trump(line[1])
                            len_hillary = self.get_len_hillary(line[1])
                            if len_trump > len_hillary:
                                label.append(1)
                            elif len_hillary > len_trump:
                                label.append(0)
                            else:
                                label.append(2)

                            tweet = self.tokenize_tweet(line[1]) # the preprocessing of the raw tweet is done here
                            tweets.append(tweet)
                    except:
                        pass

        return tweets, label


    def test(self):
        """
        call this function to check the preprocessed tweet of the test_tweet
        """
        test_tweet = "RT #Election2016 🇺🇸 ""@realDonaldTrump"" :-)))) victory :( :P #HillaryClinton 😡 or 😄 @NYSE that's 4% x &gt; 2 &amp I luv this, Makesme sooooo haaaaaapppy :) ALL OF US https://www.twitter.com/"
        tokenized_tweet = self.tokenize_tweet(test_tweet)
        print('Original tweet: ', test_tweet)
        print('Tokenized and preprocessed tweet: ', tokenized_tweet)
        print('Emojis extracted: ', self.get_emoji(test_tweet))

