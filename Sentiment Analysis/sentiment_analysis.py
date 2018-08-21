# Perform Sentiment Analysis
# This file is not completed.

import time
import gensim
import numpy as np
from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt
import re
from random import randint
import tensorflow as tf
import datetime

start_time = time.time()
print("Training Sentiment Analysis Model...")

# Load word list
model_w2v = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Vaio/Desktop/Sentiment Analysis/Word2Vec/trained_word2vec/VNESEw2v.bin', binary=True)
vocabulary = model_w2v.vocab
wordList = np.array
for word in vocabulary:
    wordList = np.append(wordList, word)
wordList = wordList.tolist()
print('Loaded word list. Length: %d' %len(wordList))

# Load word vectors
wordVectors = model_w2v.vectors
print('Loaded word vectors. Shape: ')
print(wordVectors.shape)

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

maxSeqLength = 200 # This will cover most of word in each file
numDimensions = wordVectors.shape[1] # The dimension of every word's vector

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

ids = np.load('C:/Users/Vaio/Desktop/Sentiment Analysis/Datasets/ID_matrix/VNESEidsmatrix.npy') # numFiles x maxSeqLength

# Helper functions
def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

# Implement RNN
batchSize = 32
lstmUnits = 64
numClasses = 2
iterations = 100000

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data) # Lookup for word vectos. Return batchSize x maxSeqLength x numDimensions

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.8)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "C:/Users/Vaio/Desktop/Sentiment Analysis/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# LSTM training process
print('Training LSTM...')
for i in range(iterations):
    #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch();
   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

    #Write summary to Tensorboard
   if (i % 50 == 0):
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
       writer.add_summary(summary, i)

    #Save the network every 10,000 training iterations
   if (i % 10000 == 0 and i != 0):
       save_path = saver.save(sess, "C:/Users/Vaio/Desktop/Sentiment Analysis/models/pretrained_lstm.ckpt", global_step=i)
       print("Saved to %s" % save_path)
writer.close()

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('C:/Users/Vaio/Desktop/Sentiment Analysis/models'))

iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)

print("Training time: %s seconds" % (time.time() - start_time))
