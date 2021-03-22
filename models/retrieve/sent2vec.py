import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd 

from models.retrieve.word2vec import word2vec
from src.preprocess import normalizeString


class sent2vec(word2vec):
    def __init__(self, data, scheme):
        super().__init__(data, scheme)

    def get_model(self):
        return super().get_model()

    def bow(self, sentence):
        """
        Create vectors using bag of words created from the list of words in dataset
        """
        vec = np.zeros(self.n_words)
        senctence = normalizeString(sentence)
        tokens = sentence.split()

        for i,token in enumerate(tokens):
            if word in self.words: 
                vec[i] = 1
        
        return vec


    def bow_matrix(self):
        """
        Compute embedding matrix of all questions in dataset
        """
        baseline = [len(_.split()) for _ in self.sentences]
        bow_model = np.zeros((self.n_words, len(self.sentences)))

        for i,sentence in enumerate(self.sentence):
            bow_model[:, i] = self.bow(sentence)

        return bow_model


    def tf_idf_vector(self, sentence, emb_size=40):
        """
        Create vectors using tf_idf score
        """
        vec = np.zeros(emb_size)
        senctence = normalizeString(sentence)
        tokens = sentence.split()

        for i,token in enumerate(tokens):
            if token in self.tf_dict.keys(): 
                vec[i] = self.tf_dict[token]
        
        return vec


    def tf_idf_matrix(self, emb_size=40):
        """
        Compute tf_idf vectors of questions that are in the dataset
        """
        tfidf_model = np.zeros((emb_size, len(self.sentences)))
        for i,sentence in enumerate(self.sentences):
            tfidf_model[:, i] = tf_idf_vector(sentence)

        return tfidf_model


    def seq2vec_fun(self, sentence):
        """
        Computes sequence vector using input model
        """
        senctence = normalizeString(sentence)
        tokens = sentence.split()
        len_tokens = len(tokens)
        seq_vec = np.zeros_like(self.model[:, 0], dtype=float)

        for i in range(len_tokens):
            if tokens[i] in self.words:
                index = self.words.index(tokens[i])
                seq_vec += self.model[:, index]
        
        return seq_vec/len_tokens


    def w_seq2vec_fun(self, sentence):
        """
        Computes sequence vector using input model
        """
        senctence = normalizeString(sentence)
        tokens = sentence.split()
        len_tokens = len(tokens)
        seq_vec = np.zeros_like(self.model[:, 0], dtype=float)

        for i in range(len_tokens):
            if tokens[i] in self.words:
                index = self.words.index(tokens[i])
                seq_vec += self.tf_dict[tokens[i]] * self.model[:, index]
        
        return seq_vec/len_tokens


    def seq_vec_sent(self, sentence):
        """
        Picks the sec_vec model based on model
        """

        if self.scheme == "bow":
            seq_vec = self.bow(sentence)
        if self.scheme == "tf_idf":
            seq_vec = self.tf_idf_vector(sentence)
        elif self.scheme.split('_')[0] == 'w':
            seq_vec = self.w_seq2vec_fun(sentence)
        else:
            seq_vec = self.seq2vec_fun(sentence)
        
        if 'svd' in self.scheme.split('_'):
            seq_vec = svd(seq_vec)
        if 'sketch' in self.scheme.split('_'):
            seq_vec = sketch(seq_vec)

        return seq_vec


    def get_embedding_matrix(self):
        """
        Compute embedding of questions in the corpus 
        """
        n_data = len(self.sentences)
        corpus = [' '.join([_ for _ in sent.split() if len(_)>1]) for sent in self.sentences]    
        emb_matrix = np.zeros((len(self.model[:,0]), n_data))
        
        for i,_ in enumerate(corpus):
            emb_matrix[:, i] = self.seq_vec_sent(_)
        
        return emb_matrix


if __name__ == '__main__':

    ### Read data
    path = "../Data/youssef_data.csv"
    data = pd.read_csv(path, encoding="latin-1", header=None, names=["Question","Answer"]) 
    data["Question"]=data["Question"].apply(normalizeString)
    data["Answer"]=data["Answer"].apply(normalizeString) 

    ### Init Word2vec models class
    s2v = sent2vec(data, "ws")
    model = s2v.get_embedding_matrix()

    ### embed sentences
    print(s2v.seq_vec_sent('Comment valider une compétence ?'))
