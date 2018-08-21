# Perform Sentiment Analysis

import time
import gensim
import numpy as np
from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt
import re
from random import randint

start_time = time.time()
print("Training Sentiment Analysis Model...")

maxSeqLength = 200

# Load word list
model_w2v = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Vaio/Desktop/Sentiment Analysis/Word2Vec/trained_word2vec/VNESEw2v.bin', binary=True)
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

# Count number of words in each file
print('Calculating words in each file and number of files...')
numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)       

for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)  

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))

# Draw histogram of words in file
plt.hist(numWords, 50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axis([0, 1200, 0, 8000])
plt.show()

# Text cleaning
def clean_text(text):
    text = re.sub(r'[^\w]', ' ', text) # Remove all symbols
    text = re.sub(r'^\s+' , '' , text) # Remove leading whitespace
    text = re.sub(r' +',' ', text) # Remove all duplicate white spaces
    text = text.lower()
    return text

# Reviews x maxSeqLength matrix (30000x200)
print('Training ID Matrix...')
ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
fileCounter = 0
for pf in positiveFiles:
   with open(pf, "r", encoding='utf-8') as f:
       indexCounter = 0
       line = f.readline()
       cleanedLine = clean_text(line)
       split = cleanedLine.split()
       for word in split:
           try:
               ids[fileCounter][indexCounter] = wordList.index(word)
           except ValueError:
               ids[fileCounter][indexCounter] = randint(0, len(wordList)-1) #Vector for unkown words
           indexCounter = indexCounter + 1
           if indexCounter >= maxSeqLength:
               break
       fileCounter = fileCounter + 1 

fileCounter = 0
for nf in positiveFiles:
   with open(nf, "r", encoding='utf-8') as f:
       indexCounter = 0
       line = f.readline()
       cleanedLine = clean_text(line)
       split = cleanedLine.split()
       for word in split:
           try:
               ids[fileCounter][indexCounter] = wordList.index(word)
           except ValueError:
               ids[fileCounter][indexCounter] = randint(0, len(wordList)-1) #Vector for unkown words
           indexCounter = indexCounter + 1
           if indexCounter >= maxSeqLength:
               break
       fileCounter = fileCounter + 1 

np.save('C:/Users/Vaio/Desktop/Sentiment Analysis/Datasets/ID_matrix/VNESEidsmatrix.npy', ids)
print('ID Matrix completed')
ids = np.load('C:/Users/Vaio/Desktop/Sentiment Analysis/Datasets/ID_matrix/VNESEidsmatrix.npy')

print("Training time: %s seconds" % (time.time() - start_time))

