import os
import sys
import spacy
import numpy as np

sys.path.append(os.getcwd())

from src.preprocess import normalizeString
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

nlp = spacy.load("fr_core_news_sm")


def cosine_sim(a, b):
    """
    Cosine similarity between two sets of vectors
    """
    a = np.squeeze(a) if len(a.shape)>1 else np.expand_dims(a, axis=1)
    b = np.squeeze(b) if len(b.shape)>1 else np.expand_dims(b, axis=1)
    return cosine_similarity(a.T,b.T)

def euclidian_sim(a, b):
    """
    Euclidian similarity between two sets of vectors
    """
    a = np.squeeze(a) if len(a.shape)>1 else np.expand_dims(a, axis=1)
    b = np.squeeze(b) if len(b.shape)>1 else np.expand_dims(b, axis=1)
    M = euclidean_distances(a.T,b.T)
    M = 1 - M/np.max(M)
    return M

def jaccard_sim(sentence_1, sentence_2):
    """
    Compute IoU meseaure on string tokens
    """
    sentence_1 = normalizeString(sentence_1)
    tokens_1 = sentence_1.split()
    sentence_2 = normalizeString(sentence_2)
    tokens_2 = sentence_2.split()
    
    union = np.unique(tokens_1.extend(tokens_2))
    intersection = [_ for _ in union if _ in tokens_1 and _ in tokens_2]
    return len(intersection)/len(union)

def spacy_sim(sentence_1, sentence_2):
    """
    Compute spacy similarity 
    """
    doc1 = nlp(sentence_1)
    doc2 = nlp(sentence_2)
    return doc1.similarity(doc2)


if __name__ == '__main__':
    A = np.random.rand(10,2)
    sim = cosine_sim(A[:,0], A[:,1])
    print(sim[0][0])