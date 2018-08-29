import time
import numpy as np
from os import getcwd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

start_time = time.time()
print('Training Word2Vec-LSTM Model...')

# define dataset
with open(getcwd()+'/dataset/amazon_sentiment/sentence.txt', encoding='utf-8', errors='ignore') as f:
    dataset = f.readlines()
dataset = [x.strip('\t\n') for x in dataset]
print(len(dataset))

# define label
with open(getcwd()+'/dataset/amazon_sentiment/label.txt') as f:
    train_Y = f.readlines()
train_Y = [x.strip() for x in train_Y]
train_Y = list(map(int, train_Y))
print(len(train_Y))

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(dataset)
vocab_size = len(t.word_index)+1

# integer encode the documents
encoded_dataset = t.texts_to_sequences(dataset)
max_len = 20
train_X = pad_sequences(encoded_dataset, maxlen=max_len, padding='post')

# load the whole embedding into memory
embedding_index = dict()
f = open(getcwd()+'/glove.6B.100d.txt', encoding='utf-8')
for line in f: 
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
print(len(embedding_index))
f.close()

# create a weight matrix for words in training docs
embedding_size = 100
embedding_matrix = np.zeros((vocab_size, embedding_size)) # (15,100)
for word, i in t.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
# define model
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(LSTM(64, batch_size=100))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Split train and validation set (e.g: 10000 dataset -> 8000 train, 1000 validation, 1000 test)
train_valtest = 0.8
val_test = 0.5

train_x, train_y = train_X[:int(train_valtest*len(train_X))], train_Y[:int(train_valtest*len(train_Y))]
valtest_x, valtest_y = train_X[int(train_valtest*len(train_X)):], train_Y[int(train_valtest*len(train_Y)):]

val_x, val_y = valtest_x[:int(val_test*len(valtest_x))], valtest_y[:int(val_test*len(valtest_y))]
test_x, test_y = valtest_x[int(val_test*len(valtest_x)):], valtest_y[int(val_test*len(valtest_y)):]


model.fit(train_x, train_y, validation_data = (val_x, val_y), epochs=10, verbose=1)
scores = model.evaluate(test_x, test_y, verbose=0)

print('Test on %d samples' %len(test_x))
print('Test accuracy: %.2f%%' % (scores[1]*100))

model.save(getcwd()+'/Word2Vec-LSTM_model.h5')
print('Model saved to '+getcwd()+'/Word2Vec-LSTM_model.h5')

print("Total training time: %s seconds" % (time.time() - start_time))