import tweepy
import pandas as pd
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

import sqlite3

COUNT = 0

def increment():
    global COUNT
    COUNT = COUNT+1

def setZero():
    global COUNT
    COUNT = 0

conn = sqlite3.connect('twitter.db')
print ("Opened database successfully")

conn.execute('CREATE TABLE IF NOT EXISTS flaskdata (id integer PRIMARY KEY, tweet TEXT, location TEXT, sentiment TEXT)')
conn.execute('CREATE TABLE IF NOT EXISTS tweets (id integer PRIMARY KEY, tweet TEXT, location TEXT, sentiment TEXT)')
print ("Table created successfully")

idx = 0
dummy = "val"
while idx < 52:
    conn.execute('INSERT INTO flaskdata (tweet, location, sentiment) VALUES (?,?,?)', (dummy, dummy, dummy))
    idx = idx + 1
conn.commit()
conn.close()

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
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        self.client = tweepy.API(auth)
        self.user = user

    def get_api(self):
        return self.client

class TwitterListener(tweepy.StreamListener):
    """ Basic listener class that prints received tweets """
    def on_data(self, data):
        status = json.loads(data)
        pattern = r"[Tt]rump"
        location_tweet = status['place']['full_name']
        full_tweet = ''
        if hasattr(status, 'retweeted_status') and hasattr(status.retweeted_status, 'extended_tweet'):
            full_tweet = status.retweeted_status.extended_tweet['full_text']
        elif hasattr(status, 'extended_tweet'):
            full_tweet = status['full_text']
        else:
            full_tweet = status['text']

        if (re.search(pattern, full_tweet)):
            increment()
            print(full_tweet)
            cleaned_tweet = preprocess(full_tweet)
            sentiment_tweet = predict(cleaned_tweet)['score']
            with sqlite3.connect("twitter.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO tweets (tweet, location, sentiment) VALUES (?,?,?)", (cleaned_tweet,location_tweet,sentiment_tweet))
                print("added successfully! p1")
                con.commit()
            if COUNT == 50:   
                print("TESTING TESTING TESTING")
                conn = sqlite3.connect('twitter.db')
                # conn.execute('INSERT INTO flaskdata SELECT * FROM tweets')
                conn.execute('UPDATE flaskdata set tweet=(select t.tweet from tweets t where t.id=flaskdata.id)')
                conn.execute('UPDATE flaskdata set location=(select t.location from tweets t where t.id=flaskdata.id)')
                conn.execute('UPDATE flaskdata set sentiment=(select t.sentiment from tweets t where t.id=flaskdata.id)')
                conn.execute('DELETE FROM tweets')
                setZero()
                conn.commit()

    def on_error(self, status):
        if (status == 420):
            print("error")
            # Return false on data method in case rate limit is exceeded
            return False
        print(status)

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
phoneStream = tweepy.Stream(auth  = api.auth, listener = twitter_listener, tweet_mode="extended")
phoneStream.filter(locations=[-98.7769, 29.2986, -95.3008, 33.0271])