import tweepy
import pandas as pd
import preprocessor as p
import nltk
import keras
import re
import string
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Word2vec
import gensim
from gensim.models import Word2Vec

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

consumer_key = "eYidH9OOsZyXRGl4Gbx9nJaY7"
consumer_secret = "35qEXiH8XoSEfEHjc1yHyeuYitwINwlspDtjFFur0KjkZ4YnDt"
access_token = "1247950189270380544-JR9xdqYe9eJrwt3hr9S6s1f8N31sLn"
access_secret = "MXMk4PGkzwuOiJGRbPxVHI3j2tCNvF6RxgRzQjVhGS1S8"

# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 5
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"

dF2 = pd.read_csv('./train_test_data.csv', encoding =DATASET_ENCODING , names=DATASET_COLUMNS, skiprows = 1)
dF2_train, dF2_test = train_test_split(dF2, test_size=1-TRAIN_SIZE, random_state=42)

tokenizer = Tokenizer()
train_list = [str(text) for text in dF2_train.text]
test_list = [str(text) for text in dF2_test.text]
tokenizer.fit_on_texts(train_list)

model = keras.models.load_model('model.h5')

def type_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = type_sentiment(score, include_neutral=include_neutral)
    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}  

class TwitterClient():
    def __init__(self, user=None):
        self.auth = TwitterAuthenticator().authenticate_app()
        self.client = tweepy.API(self.auth)
        self.user = user

    def get_api(self):
        return self.client

    def get_timeline_tweets(self, num_tweets):
        tweets = []
        list_tweets = tweepy.Cursor(self.client.user_timeline, id = self.user).items(num_tweets)
        for tweet in list_tweets:
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in tweepy.Cursor(self.client.friends, id = self.user).items(num_friends):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline(self, num_tweets):
        home_timeline_tweets = []
        for tweet in tweepy.Cursor(self.client.home_timeline, id = self.user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets

class TwitterAuthenticator():
    def authenticate_app(self):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        return auth

class TwitterStreamer():
    """ Class for streaming and processing live tweets """
    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()

    def stream_tweets(self, filename, hashtag_list):
        # Handles Twitter authentication and connection to Streaming API
        listener = TwitterListener(filename)
        auth = self.twitter_authenticator.authenticate_app()
        api = tweepy.API(auth)
        stream = tweepy.Stream(auth = api.auth, listener = listener)
        # stream.filter(track = ["donald trump", "hillary clinton", "barack obama", "bernie sanders"])
        stream.filter(track = hashtag_list)

class TwitterListener(tweepy.StreamListener):
    """ Basic listener class that prints received tweets """
    def on_data(self, data):
        status = json.loads(data)
        pattern = r"[Tt]rump"
        # if (not hasattr(status, "retweeted_status")) and 'RT @' not in status['text']:  # Check if Retweet
        try:
            if (re.search(pattern, status.extended_tweet["full_text"])):
                print(status['place']['full_name'])
                print(status.extended_tweet["full_text"])
                # print(status)
        except AttributeError:
            if  (re.search(pattern, status['text'])):
                print(status['place']['full_name'])
                print(status['text'])
                # print(status)

    def on_error(self, status):
        if (status == 420):
            print("error")
            # Return false on data method in case rate limit is exceeded
            return False
        print(status)

class TweetAnalyzer():
    """ Functionality for analyzing and categorizing content from tweets """
    def tweets_to_dataframe(self, tweets):            
        df = pd.DataFrame(data = [tweet.full_text for tweet in tweets], columns = ["tweets"])
        df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.full_text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
        return df

def preprocess(text):
    stop_words = stopwords.words("english")
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    lemmatizer = WordNetLemmatizer()
    for token in text.split():
        if token not in stop_words:
            tokens.append(lemmatizer.lemmatize(token))
    return " ".join(tokens)

client = TwitterClient()
api = client.get_api()
twitter_listener = TwitterListener()
phoneStream = tweepy.Stream(auth  = api.auth, listener = twitter_listener)
phoneStream.filter(locations=[-98.7769, 29.2986, -95.3008, 33.0271])