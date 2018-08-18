import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Vaio/Desktop/Sentiment Analysis/Word2Vec/trained_word2vec/VNTQw2v.bin', binary=True)

def display_closestwords_tsne(model, word, num_sim):
    array = np.empty((0,300), dtype='f')
    word_labels = [word]
    closest_words = model.similar_by_word(word, topn = num_sim)
    
    for word_score in closest_words:
        word_vector = model[word_score[0]]
        word_labels.append(word_score[0])
        array = np.append(array, np.array([word_vector]), axis=0)
    
    tsne = TSNE(n_components = 2)
    X = tsne.fit_transform(array)
    x_coords = X[:, 0]
    y_coords = X[:, 1]
    
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
    
word = input("Enter word: ")
num_sim = int(input("Number of similar words: "))
display_closestwords_tsne(model, word, num_sim)