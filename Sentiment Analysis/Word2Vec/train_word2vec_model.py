# Vietnam Word2Vec Model 

import os
#import sys
import codecs
import gensim
import time

start_time = time.time()

class MySentences(object):
     def __init__(self, dirname):
         self.dirname = dirname

     def __iter__(self):
         for fname in os.listdir(self.dirname):
             for line in codecs.open(os.path.join(self.dirname, fname), 'r', 'utf-8'):
                 yield line.split()

dirData='C:/Users/Vaio/Desktop/Sentiment Analysis/Word2Vec/Tokenizer/VNTQcorpus_small_tok'
pathModelBin='C:/Users/Vaio/Desktop/Sentiment Analysis/Word2Vec/trained_word2vec/VNTQw2v.bin'
pathModelTxt='C:/Users/Vaio/Desktop/Sentiment Analysis/Word2Vec/trained_word2vec/VNTQw2v.txt'

if __name__ == '__main__':
    sentences = MySentences(dirData) # a memory-friendly iterator
    
    print("Training Word2Vec model...")
    model = gensim.models.Word2Vec(sentences, size=300, window=10, min_count=10, sample=0.0001, workers=4, sg=0, negative=10, cbow_mean=1, iter=5)
    model.wv.save_word2vec_format(pathModelBin, fvocab=None, binary=True)
    model.wv.save_word2vec_format(pathModelTxt, fvocab=None, binary=False)
    print("Training completed")
    print("Training time: %s seconds" % (time.time() - start_time))