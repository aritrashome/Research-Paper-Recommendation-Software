import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords 
import re
from nltk.stem import PorterStemmer
from collections import Counter
import re

df = pd.read_csv('data.csv')
with open("Topic/test.txt", "rb") as fp:   # Unpickling
    w2vec_lda = pickle.load(fp)
model = Word2Vec.load("Topic/word2vec.model")

df['Topic'] = 0

def topic_assign(string):
    string = string.lower()
    string = re.sub(r'\W+', ' ', string)
    string = word_tokenize(string)
    ps = PorterStemmer()
    string = [ps.stem(w) for w in string]
    #string
    res = np.zeros(n_topics)
    from sklearn.metrics.pairwise import cosine_similarity
    for w in string:
        if w not in model.wv.vocab:
            continue
        sim = np.zeros(n_topics)
        for j in range(len(w2vec_lda)):
            for k in range(len(w2vec_lda[j])):
                sim[j] = sim[j] + cosine_similarity( model.wv[w2vec_lda[j][k][0]].reshape(1,50), model.wv[w].reshape(1,50) )[0][0]*w2vec_lda[j][k][1]
        res[sim.argmax()] = res[sim.argmax()] + 1
    if res[res.argmax()]==0:
        return None
    return res.argmax() + 1

for i in range(len(df)):
	df['Topic'][i] = topic_assign(df['Title'][i])
print(df)