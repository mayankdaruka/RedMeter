import tweepy
import pandas as pd
import sys
import csv
import matplotlib.pyplot as plt
from textblob import TextBlob
import preprocessor as p
import nltk
import keras
import string
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

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
import pickle
import itertools

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

dataset_path = './training.1600000.processed.noemoticon.csv'
dF = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
print(dF.head(5))
print(dF.shape[0])
print("Dataset Size: " + len(dF))
decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

def decode_sentiment(label):
    return decode_map[int(label)]

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

dF.target = dF.target.apply(lambda x: decode_sentiment(x))
dF.text = dF.text.apply(lambda x: preprocess(x))
dF.to_csv(r'train_test_data.csv', index=False)
dF2 = pd.read_csv('./train_test_data.csv', encoding =DATASET_ENCODING , names=DATASET_COLUMNS, skiprows = 1)
dF2_train, dF2_test = train_test_split(dF2, test_size=1-TRAIN_SIZE, random_state=42)
print("Train Size: " + str(len(dF2_train)))
print("Test Size: " + str(len(dF2_test)))
print(dF2_train.text)
documents = [str(text).split() for text in dF2_train.text]
w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 
                                            window=W2V_WINDOW, 
                                            min_count=W2V_MIN_COUNT, 
                                            workers=8)
w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()
vocab_size = len(words)
print("Vocab size", vocab_size)
w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
w2v_model.save(WORD2VEC_MODEL)

w2v_model = Word2Vec.load('model.w2v')
print(w2v_model.most_similar("love"))
tokenizer = Tokenizer()
train_list = [str(text) for text in dF2_train.text]
test_list = [str(text) for text in dF2_test.text]
tokenizer.fit_on_texts(train_list)
voc_size = len(tokenizer.word_index) + 1
print("Total words: ", voc_size)

labels = dF2_train.target.unique().tolist()
labels.append(NEUTRAL)
print(labels)

encoder = LabelEncoder()
encoder.fit(dF2_train.target.tolist())

y_train = encoder.transform(dF2_train.target.tolist())
y_test = encoder.transform(dF2_test.target.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

x_train = pad_sequences(tokenizer.texts_to_sequences(train_list), maxlen=SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_list), maxlen=SEQUENCE_LENGTH)

print("x_train", x_train.shape)
print("y_train", y_train.shape)
print()
print("x_test", x_test.shape)
print("y_test", y_test.shape)

embedding_matrix = np.zeros((voc_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)
embedding_layer = Embedding(voc_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)

model.save(KERAS_MODEL)