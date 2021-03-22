import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from src.preprocess import normalizeString
from models.retrieve.sent2vec import sent2vec
from models.retrieve.similarities import cosine_sim, euclidian_sim


def answer_question(question, s2v_model, data, similarity, k=1):
    """
    Uses vector representation and similarity function 
    to provide K possible answers to the question
    """
    # Clean question
    question = normalizeString(question)
    embed_matrix = s2v_model.get_embedding_matrix()
    query_vec = s2v_model.seq_vec_sent(question)

    if query_vec.shape[0] == embed_matrix.shape[0]:
        
        # Compute similarity
        X = similarity(query_vec, embed_matrix)[0]

        # Get best matchs indexes'
        indexes = np.argsort(X)

        # Extract responses from data
        Y = data.iloc[indexes, 1].drop_duplicates('last')
        Y_indexes = list(Y.index)
        responses = Y.to_numpy()[-k:]
        similarities = [X[Y_indexes[-k+i]] for i in range(len(responses))]

    else:
        return('seqvec embedding dimensions {} \n'\
                'model embedding dimensions {}'.format(query_vec.shape, embed_questions.shape))

    return responses, similarities


if __name__ == '__main__':
    
    ### Read data
    path = "../Data/youssef_data.csv"
    data = pd.read_csv(path, encoding="latin-1", header=None, names=["Question","Answer"]) 
    data["Question"] = data["Question"].apply(normalizeString)
    data["Answer"] = data["Answer"].apply(normalizeString) 

    ### Select model
    s2v = sent2vec(data, 'ws')

    ### Answer question
    if True:
        input_question = input(">>>")
        while input_question not in ['quit', 'q']: 
            answers, sims = answer_question(input_question, s2v, data, cosine_sim, 10)
            for i, rep in enumerate(answers):
                print("\t >>> Similarity : {} - {}".format(float(sims[i]), rep))

            input_question = input(">>>")

        
