import os
import sys
import nltk
import spacy
import torch
import warnings
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from models.retrieve.utils import *
from src.preprocess import normalizeString
from gensim.models import KeyedVectors, Word2Vec

warnings.filterwarnings(action='ignore')
nlp = spacy.load("fr_core_news_sm")

class word2vec():
    def __init__(self, data, scheme='ws'):
        self.data = data
        self.qa_corpus = qa_corpus(data)
        self.sentences = sentences(data)
        self.words = words(data)
        self.n_words = len(self.words)
        self.tf_dict = tf_table(self.sentences)
        self.scheme = scheme
        self.model = self.get_model()
        

    def get_embedding_list(self):
        s = "There are multiple embedding kernels : "
        embs = ["", "fauconnier", "BERT", "OneHot", "W-S Matrix", "Cooccurences Matrix", "CBOW", "Skip-gram"]
        return s + "\n -".join(embs)


    def fauconnier(self, cat='cbow'):
        """
        Compute word2vec matrix for all words in corpus using fauconnier embedding
        """
        model_name = 'frWiki_no_lem_no_postag_no_phrase_1000_' + cat + '_cut200.bin'
        model_path = os.path.join(os.getcwd(), "models", "retrieve", "models", model_name)
        fauconnier = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors="ignore")
        fauconnier_model = np.zeros((1000, self.n_words))

        for i,word in enumerate(self.words):
            try:
                fauconnier_model[:, i] = fauconnier.wv[word]
            except:
                fauconnier_model[:, i] = np.zeros(1000)

        return fauconnier_model

    
    def camembert(self):
        """
        Compute word2vec matrix for all words in corpus using BERT embedding
        """
        camembert = torch.hub.load('pytorch/fairseq', 'camembert')
        camem_model = np.zeros((768, self.n_words))
        for i,word in enumerate(self.words):
            try:
                tokens = camembert.encode(word)
                features = camembert.extract_features(tokens)[0][0]
                camem_model[:, i] = features.detach().numpy()
            except:
                camem_model[:, i] = np.zeros(768)

        return camem_model

    
    def spacy_emb(self):
        """
        Spacy embedding based on the tok2vec model pretrained on news data 
        """
        spacy_model = np.zeros((96, self.n_words))
        
        for i,word in enumerate(self.words):
            try :
                doc = nlp(word)
                features = np.squeeze(doc[0].vector.reshape(1,-1))
                spacy_model[:, i] = features
            except:
                spacy_model[:, i]  = np.zeros(96)

        return spacy_model


    def one_hot(self):
        """
        One hot encoding of words in corpus
        """
        onehot_model = np.eye(self.n_words)
        return onehot_model

    
    def word_sentence(self, weighted=False, reduce=None):
        """
        Compute Word-sentence matrix
        """
        n = len(self.sentences)
        ws_model = np.zeros((n,self.n_words))

        for i,sentence in enumerate(self.sentences):
            for j,word in enumerate(self.words):
                if word in sentence:
                    if weighted:
                        if word in self.tf_dict.keys():
                            ws_model[i,j] += self.tf_dict[word]
                        else:
                            print("word ignored in tf_idf : ", word)
                    else:
                        ws_model[i,j] += 1

        if reduce == "svd":
            ws_model = svd(ws_model)

        if reduce == "sketch":
            ws_model = sketch(ws_model)
            
        return ws_model    


    def cooccurences(self, window=2, reduce=None):
        """
        Compute cooccurances matrix
        """
        cooc_model = np.zeros((self.n_words, self.n_words))
        tokenized_sent = [_.split() for _ in self.sentences]    
        tokenized_sent = [[_ for _ in sent if len(_)>1]  for sent in tokenized_sent]    
        window_list = lambda l,i,w: l[min(i-w, 0):min(len(l)-1, i+w)]
            
        for i, sent in enumerate(tokenized_sent):
            for j, word_1 in enumerate(sent):
                wind = window_list(sent, j, window)
                index_1 = self.words.index(word_1)
                for word_2 in wind:
                    index_2 = self.words.index(word_2)
                    cooc_model[index_1,index_2] += 1*(index_1!=index_2)
                    cooc_model[index_2,index_1] += 1*(index_1!=index_2)
        
        if reduce == "svd":
            cooc_model = svd(cooc_model)

        if reduce == "sketch":
            cooc_model = sketch(cooc_model)

        return cooc_model


    def sgram(self, emb_dim=100):
        """
        Compute word embedding using Skip gram model from gensim
        """
        corpus = [_.split() for _ in self.sentences]
        sgram = Word2Vec(sentences=corpus, size=emb_dim, min_count=2, sg=1, hs=1)
        sgram_model = np.zeros((emb_dim, self.n_words))
        
        for i,word in enumerate(self.words):
            try:
                sgram_model[:, i] = sgram.wv[word]
            except:
                pass
        
        return sgram_model


    def cbow(self, emb_dim=100):
        """
        Compute word embedding using CBOW model from gensim
        """
        corpus = [_.split() for _ in self.sentences]
        cbow = Word2Vec(sentences=corpus, size=emb_dim, min_count=2, sg=0, hs=1)
        cbow_model = np.zeros((emb_dim, self.n_words))
        
        for i,word in enumerate(self.words):
            try:
                cbow_model[:, i] = cbow.wv[word]
            except:
                pass
        
        return cbow_model


    def get_model(self):
        """
        Returns all models computed in dict format
        """
        if self.scheme == 'fauc_cbow'   : model = self.fauconnier('cbow')
        if self.scheme == 'fauc_sgram'  : model = self.fauconnier('skip')
        if self.scheme == 'camembert'   : model = self.camembert()
        if self.scheme == 'spacy_emb'   : model = self.spacy_emb()
        if self.scheme == 'one_hot'     : model = self.one_hot()
        if self.scheme == 'ws'          : model = self.word_sentence(weighted=False, reduce=None)
        if self.scheme == 'w_ws'        : model = self.word_sentence(weighted=True,  reduce=None)
        if self.scheme == 'ws_svd'      : model = self.word_sentence(weighted=False, reduce='svd')
        if self.scheme == 'w_ws_svd'    : model = self.word_sentence(weighted=True,  reduce='svd')
        if self.scheme == 'ws_sketch'   : model = self.word_sentence(weighted=False, reduce='sketch')
        if self.scheme == 'w_ws_sketch' : model = self.word_sentence(weighted=True,  reduce='sketch')
        if self.scheme == 'cooc_2'      : model = self.cooccurences()
        if self.scheme == 'cooc_5'      : model = self.cooccurences(window=5)
        if self.scheme == 'cooc_svd'    : model = self.cooccurences(reduce='svd')
        if self.scheme == 'cooc_sketch' : model = self.cooccurences(reduce='sketch')
        if self.scheme == 'sgram'       : model = self.sgram()
        if self.scheme == 'cbow'        : model = self.cbow()    
        return model



if __name__ == '__main__':

    ### Read data
    path = "../Data/youssef_data.csv"
    data = pd.read_csv(path, encoding="latin-1", header=None, names=["Question","Answer"]) 
    
    ### Init Word2vec models class
    w2v = word2vec(data, 'ws')

    ### List of embdeddings
    print(w2v.get_embedding_list())

    ### Select CBOW model
    model = w2v.get_model()

    ### Plot word embeddings of words in corpus
    print(model)
    plt.imshow(model)
    plt.show()
