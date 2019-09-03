# coding=utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

corpus = [line.strip() for line in open('')]

vectorizer = TfidfVectorizer(analyzer='word', token_pattern='(?u)\\b\\w+\\b')

vectorizer.fit(corpus)

idx = 0
vocab = {}
idf_vector = []
with open('vocab_idf', 'w') as f:
  for word, idf in zip(vectorizer.get_feature_names(), vectorizer.idf_):
    f.write('{}\t{}\n'.format(word, idf))
    vocab[word] = idx
    idf_vector.append(idf)
    idx += 1

path = ''
data = [line.strip() for line in open(path)]

vector = []
for s in data:
  ss = s.split(' ')
  v_idf = np.array(idf_vector)
  v_tf = np.zeros(v_idf.shape)
  for c in ss:
    if c in vocab:
      v_tf[vocab[c]] = ss.count(c) / len(ss)
  v_tfidf = v_tf * v_idf
  vector.append(v_tfidf)

np.save('tfidf_vector.npy', np.asarray(vector)) 
