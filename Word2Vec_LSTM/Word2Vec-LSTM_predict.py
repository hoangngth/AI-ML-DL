import numpy as np
from os import getcwd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

model = load_model(getcwd()+'/Word2Vec-LSTM_model.h5')

with open(getcwd()+'/dataset/amazon_sentiment/sentence.txt', encoding='utf-8', errors='ignore') as f:
    dataset = f.readlines()
dataset = [x.strip('\t\n') for x in dataset]

t = Tokenizer()
t.fit_on_texts(dataset)

max_len = 20
user_sentence = input('Enter your sentence: ')
temp_sentence = []
for word in user_sentence.split(' '):
    if word not in t.word_index:
        continue
    temp_sentence.append(t.word_index[word])
    temp_sentence_padded = pad_sequences([temp_sentence], maxlen=max_len) 
print('Sentiment: ',(model.predict(np.array([temp_sentence_padded][0]))[0][0]))