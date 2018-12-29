#!/usr/bin/env python

#  Author: Parveen kumar
#  Date: 29/12/2018
#
# *************************************** #

import pandas as pd
import numpy as numpy
import logging
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from bs4 import BeautifulSoup
import os, re
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

def review_to_wordlist(review, tokenizer, remove_stopwords=False):
    
    review_text = BeautifulSoup(review).get_text()

    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return words

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences
    
print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)
    
 logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print("Training Word2Vec model")
model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling, seed=1)

# The obtained word2vec model is now trained

pretrained_weights = w2vModel.syn0

# Now create LSTM model with embedding later with pretrained weights
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=emdedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

def word2idx(word):
  return word_model.wv.vocab[word].index

def idx2word(idx):
  return word_model.wv.index2word[idx]

train_x = np.zeros([25000, 2500], dtype=np.int32)
train_y = np.zeros([25000, 1], dtype=np.int32)

i = 0
j = 0

for review in train["review"]:
	review_text = BeautifulSoup(review).get_text()
	review_text = re.sub("[^a-zA-Z]", " ", review_text)
	words = review_text.lower().split()
	j=0
	for x in words:
		x in word_vectors.vocab:
			train_x[i][j] = word2idx(x)
				j += 1
	i += 1

i = 0

for sentiment in train.sentiment:
  train_y[i] = sentiment
  i += 1

def on_epoch_end(epoch, _):
  print('\nGenerating text after epoch: %d' % epoch)
  texts = [
    'deep convolutional',
    'simple and effective',
    'a nonconvex',
    'a',
  ]
  for text in texts:
    sample = generate_next(text)
    print('%s... -> %s' % (text, sample))

model.fit(train_x, train_y,
          batch_size=128,
          epochs=20,
          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
