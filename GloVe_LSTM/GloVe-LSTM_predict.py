import numpy as np
from os import getcwd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

model = load_model(getcwd()+'GloVe-LSTM_model.h5')

dataset = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']

t = Tokenizer()
t.fit_on_texts(dataset)

max_len = 4
user_sentence = input('Enter your sentence: ')
temp_sentence = []
for word in user_sentence.split(' '):
    temp_sentence.append(t.word_index[word])
    temp_sentence_padded = pad_sequences([temp_sentence], maxlen=max_len) 
print('Sentiment: ',(model.predict(np.array([temp_sentence_padded][0]))[0][0]))