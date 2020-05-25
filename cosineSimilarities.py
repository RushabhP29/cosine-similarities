from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def plot_mywords(words):
    plt.style.use('bmh')
    word_vectors = np.vstack([model[w] for w in words])
    word_vectors.shape
    twodim = PCA().fit_transform(word_vectors)[:, :2]
    twodim.shape
    plt.figure(figsize=(15, 15))
    plt.scatter(twodim[:, 0], twodim[:, 1], c='r', marker=(5, 2))
    for word, (x, y) in zip(words, twodim):
        plt.text(x, y, word, fontsize=10, fontname='Courier', fontweight='normal')

    plt.axis('on')
    plt.draw()
    plt.show()


print('')
print('-->Connecting to Word2Vec')
path ='bin'
model= KeyedVectors.load(path)
print('')
print('Done Loading')
print('Number of words: ', len(model.wv.vocab))

targetword ='good'

if targetword in model.wv.vocab:
    print('Input word exits in the vocabulary')
    words_list = []
    similarwords = model.most_similar([targetword],topn=30)
    for i in similarwords:
        print(i[0],i[1])
        words_list.append(i[0])
    print('-----------------------------------------')
else:
    print('The Input word does not exist in the model vocab')

plot_mywords(words_list)
