# Perform Sentiment Analysis

import gensim
import numpy as np
from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt

# Load word list
model_w2v = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Vaio/Desktop/Sentiment Analysis/Word2Vec/trained_word2vec/VNTQw2v.bin', binary=True)
vocabulary = model_w2v.vocab
wordList = np.array
for word in vocabulary:
    wordList = np.append(wordList, word)
wordList = wordList.tolist()
print('Loaded word list. Length: %d' %len(wordList))

# Load word vectors
wordVector = model_w2v.vectors
print('Loaded word vectors. Shape: ')
print(wordVector.shape)

# Load negative and positive files
dirPos = 'C:/Users/Vaio/Desktop/Sentiment Analysis/Datasets/data_train/train/pos/'
dirNeg = 'C:/Users/Vaio/Desktop/Sentiment Analysis/Datasets/data_train/train/neg/'
positiveFiles = [dirPos + f for f in listdir(dirPos) if isfile(join(dirPos, f))]
negativeFiles = [dirNeg + f for f in listdir(dirNeg) if isfile(join(dirNeg, f))]


