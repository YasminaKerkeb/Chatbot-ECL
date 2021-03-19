import sklearn

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def words(data):
    """
    read dataFrame data and concatenate all entries
    """
    words_list = data["Question"].map(str) + " " +  data["Answer"].map(str) 
    words_list = " ".join(words_list.to_numpy()).split()
    words_list = np.unique(words_list)
    words_list = [_ for _ in words_list if len(_)>1]
    return list(words_list)

def qa_corpus(data):
    """
    read DataFrame and create list of sentences (list of tokens)
    """
    x = data["Question"].to_numpy()
    y = data["Answer"].to_numpy()
    result = list(np.hstack([x,y]))
    return result

def sentences(data):
    """
    read dataFrame data and concatenate all entries
    """
    corpus = data["Question"].map(str) + " " +  data["Answer"].map(str) 
    return corpus.to_numpy()

def tf_idf_table(sentences):
    """
    Compute tf_idf index from given corpus
    """
    # sentences = [" ".join(sentences)]
    tf_idf = TfidfVectorizer()
    tf_idf_fitted = tf_idf.fit_transform(sentences)
    feature_names = tf_idf.get_feature_names()
    
    tf_idf_dict = {}
    for col in tf_idf_fitted.nonzero()[1]:
        tf_idf_dict[feature_names[col]] = tf_idf_fitted[0, col]
        
    return tf_idf_dict

def tf_table(sentences):
    """
    Compute tf_idf index from given corpus
    """
    corpus = " ".join(sentences).split()
    n_corpus = len(corpus)
    tf_dict = dict(zip(corpus, np.zeros(n_corpus)))
    
    for word in corpus:
        tf_dict[word] += 1/n_corpus
        
    return tf_dict

def svd(M, target_dim=50):
    """
    Preform truncated SVD to reduce dimensionality
    """
    svd = TruncatedSVD(n_components=50)
    N = svd.fit_transform(M.T).T
    return N

def sketch(M, target_dim=50):
    """
    Compute sketch of data to reduce dimensionality
    """
    d,n = M.shape[:2]
    W = np.random.rand(d, target_dim)
    S = np.cos(np.dot(W.T,M))
    return S/n 
