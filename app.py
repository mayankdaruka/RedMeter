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

import sqlite3

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

from flask import Flask, render_template, url_for
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    houston_tweets = []
    dallas_tweets = []
    fortworth_tweets = []
    sanantonio_tweets = []
    austin_tweets = []
    conn = sqlite3.connect('twitter.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM flaskdata WHERE location="Houston, TX"') 
    rows = cur.fetchall()
    for row in rows:
        houston_tweets.append(float(row[3]))
    cur.execute('SELECT * FROM flaskdata WHERE location="The Woodlands, TX"') 
    rows = cur.fetchall()
    for row in rows:
        houston_tweets.append(float(row[3]))
    cur.execute('SELECT * FROM flaskdata WHERE location="Pearland, TX"') 
    rows = cur.fetchall()
    for row in rows:
        houston_tweets.append(float(row[3]))
    cur.execute('SELECT * FROM flaskdata WHERE location="Pasadena, TX"') 
    rows = cur.fetchall()
    for row in rows:
        houston_tweets.append(float(row[3]))
    cur.execute('SELECT * FROM flaskdata WHERE location="Cinco Ranch, TX"') 
    rows = cur.fetchall()
    for row in rows:
        houston_tweets.append(float(row[3]))
    cur.execute('SELECT * FROM flaskdata where location="Dallas, TX"')
    rows = cur.fetchall()
    for row in rows:
        dallas_tweets.append(float(row[3]))
    cur.execute('SELECT * FROM flaskdata where location="Richardson, TX"')
    rows = cur.fetchall()
    for row in rows:
        dallas_tweets.append(float(row[3]))
    cur.execute('SELECT * FROM flaskdata where location="Arlington, TX"')
    rows = cur.fetchall()
    for row in rows:
        dallas_tweets.append(float(row[3]))
    cur.execute('SELECT * FROM flaskdata where location="Plano, TX"')
    rows = cur.fetchall()
    for row in rows:
        dallas_tweets.append(float(row[3]))
    cur.execute('SELECT * FROM flaskdata where location="San Antonio, TX"')
    rows = cur.fetchall()
    for row in rows:
        sanantonio_tweets.append(float(row[3]))
    cur.execute('SELECT * FROM flaskdata where location="Austin, TX"')
    rows = cur.fetchall()
    for row in rows:
        austin_tweets.append(float(row[3]))
    cur.execute('SELECT * FROM flaskdata where location="Fort Worth, TX"')
    rows = cur.fetchall()
    for row in rows:
        fortworth_tweets.append(float(row[3]))
    hou_mean = np.mean(houston_tweets)
    dal_mean = np.mean(dallas_tweets)
    san_mean = np.mean(sanantonio_tweets)
    for_mean = np.mean(fortworth_tweets)
    aus_mean = np.mean(austin_tweets)
    hou_nan = np.isnan(hou_mean)
    dal_nan = np.isnan(dal_mean)
    san_nan = np.isnan(san_mean)
    for_nan = np.isnan(for_mean)
    aus_nan = np.isnan(aus_mean)
    return render_template('index.html', hou=hou_mean, dal=dal_mean, fort=for_mean, san=san_mean, aus=aus_mean, hou_nan=hou_nan, san_nan=san_nan, aus_nan=aus_nan, for_nan=for_nan, dal_nan=dal_nan)

if __name__ == '__main__':
    app.run(debug=True)
