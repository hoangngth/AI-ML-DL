# Perform Sentiment Analysis

import time
import gensim
import numpy as np
import os
from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt
import re
from random import randint
import tensorflow as tf

maxSeqLength = 200 # This will cover most of word in each file
start_time = time.time()
print("Training Sentiment Analysis Model...")

# Load word list
model_w2v = gensim.models.KeyedVectors.load_word2vec_format(os.getcwd()+'/word2vec/trained_word2vec/VNESEw2v.bin', binary=True)
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
dirPos = os.getcwd()+'/dataset/data_train/train/pos/'
dirNeg = os.getcwd()+'/dataset/data_train/train/neg/'
positiveFiles = [dirPos + f for f in listdir(dirPos) if isfile(join(dirPos, f))]
negativeFiles = [dirNeg + f for f in listdir(dirNeg) if isfile(join(dirNeg, f))]

# Count number of words in each file
print('Calculating words in each file and number of files...')
numWords = []
labels = []
numPF, numNF = 0, 0
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)
        labels.append(1)
        numPF += 1

for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)  
        labels.append(0)
        numNF += 1

labels = np.array(labels)
numFiles = len(numWords)
print('The total number of positive file is', numPF)
print('The total number of positive file is', numNF)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))
print('Finished labeling %d files' %len(labels))

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
print('Training Feature Matrix...')
features = np.zeros((numFiles, maxSeqLength), dtype='int32')
fileCounter = 0
for pf in positiveFiles:
   with open(pf, "r", encoding='utf-8') as f:
       indexCounter = 0
       line = f.readline()
       cleanedLine = clean_text(line)
       split = cleanedLine.split()
       for word in split:
           try:
               features[fileCounter][indexCounter] = wordList.index(word)
           except ValueError:
               features[fileCounter][indexCounter] = randint(0, len(wordList)-1) #Vector for unkown words
           indexCounter = indexCounter + 1
           if indexCounter >= maxSeqLength:
               break
       fileCounter = fileCounter + 1 

for nf in negativeFiles:
   with open(nf, "r", encoding='utf-8') as f:
       indexCounter = 0
       line = f.readline()
       cleanedLine = clean_text(line)
       split = cleanedLine.split()
       for word in split:
           try:
               features[fileCounter][indexCounter] = wordList.index(word)
           except ValueError:
               features[fileCounter][indexCounter] = randint(0, len(wordList)-1) #Vector for unkown words
           indexCounter = indexCounter + 1
           if indexCounter >= maxSeqLength:
               break
       fileCounter = fileCounter + 1 

np.save(os.getcwd()+'/dataset/features_matrix/VNESEfeaturesmatrix.npy', features)
print('Feature Matrix completed')

features = np.load(os.getcwd()+'/dataset/features_matrix/VNESEfeaturesmatrix.npy') # numFiles x maxSeqLength

# Training set and Validation set splitting
split_fractor = 0.8
split_index = int(split_fractor * len(features)) # 24000
splitter = len(features)//2 # 15000

train_x_pos, val_x_pos = features[:(split_index//2)], features[(split_index//2):splitter] # :12000, 12000:15000
train_x_neg, val_x_neg = features[splitter:int(split_index + splitter * (1-split_fractor))], features[int(split_index + splitter * (1-split_fractor)):] # 15000:(24000+3000), (24000+3000): 
train_x, val_x = np.append(train_x_pos, train_x_neg, axis=0), np.append(val_x_pos, val_x_neg, axis=0)

train_y_pos, val_y_pos = labels[:(split_index//2)], labels[(split_index//2):splitter] # :12000, 12000:15000
train_y_neg, val_y_neg = labels[splitter:int(split_index + splitter * (1-split_fractor))], labels[int(split_index + splitter * (1-split_fractor)):] # 15000:(24000+3000), (24000+3000): 
train_y, val_y = np.append(train_y_pos, train_y_neg), np.append(val_y_pos, val_y_neg)

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape))
print("Label set: \t\t{}".format(train_y.shape), 
      "\nValidation label set: \t{}".format(val_y.shape))

lstm_units = 20
lstm_layers = 1
batch_size = 50
learning_rate = 0.001
n_words = len(wordList)
embed_size = wordVectors.shape[1] # The dimension of every word's vector

# Create Graph Object
tf.reset_default_graph()
with tf.name_scope('input'):
    inputs_ = tf.placeholder(tf.int32, [None, None], name = 'input')
    labels_ = tf.placeholder(tf.int32, [None, None], name = 'label')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    
with tf.name_scope('embeddings'):
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)
    
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell
    
with tf.name_scope("RNN_layers"):
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for layer in range(lstm_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    
with tf.name_scope("RNN_forward"):
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
    
with tf.name_scope('predictions'):
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.nn.softsign) # Sigmoid/Tanh/Softsign
    tf.summary.histogram('predictions', predictions)
with tf.name_scope('cost'):
    cost = tf.losses.mean_squared_error(labels_, predictions)
    tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

merged = tf.summary.merge_all()

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
def get_batches(x, y, batch_size):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

epochs = 5

# with graph.as_default():
saver = tf.train.Saver()

print('Training LSTM model...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(os.getcwd()+'/logs/train', sess.graph)
    test_writer = tf.summary.FileWriter(os.getcwd()+'/logs/test', sess.graph)
    iteration = 1
    for e in range(epochs):
        train_acc = []
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            summary, batch_acc, loss, state, _ = sess.run([merged, accuracy, cost, final_state, optimizer], feed_dict=feed)
            train_acc.append(batch_acc)
            
            train_writer.add_summary(summary, iteration)
        
            if iteration%5==0:
                print("Epoch: {}/{} |".format(e, epochs),
                      "Iteration: {} |".format(iteration),
                      "Train loss: {:.3f} |".format(loss),
                      "Training accuracy: {:.3f}".format(np.mean(train_acc)))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}

                    summary, batch_acc, val_state = sess.run([merged, accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Validation accuracy: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
            test_writer.add_summary(summary, iteration)
            saver.save(sess, os.getcwd()+'/checkpoint/sentiment.ckpt')
    saver.save(sess, os.getcwd()+'/checkpoint/sentiment.ckpt')

print("Total training time: %s seconds" % (time.time() - start_time))
