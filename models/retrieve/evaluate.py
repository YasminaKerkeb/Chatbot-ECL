import os
import sys
import language_check
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.retrieve.similarities import *
from models.retrieve.retrieve import answer_question
from models.retrieve.sent2vec import sent2vec

tool = language_check.LanguageTool('fr-FR')


def evaluate_answer(question, answer, bot_answer):
    """
    Evaluate the goodness of an answer
    """
    qa = question + " " + answer
    qa_pred = question + " " + bot_answer

    sims = {'jaccard' : jaccard_sim(qa, qa_pred),
            'spacy'   : spacy_sim(qa, qa_pred)}
    return sims  


def evaluate_answer_w2v(question, answer, bot_answer, s2v_model):
    """
    Evaluate goodness of answer using cosine and euclidian similarity
    """
    qa = question + " " + answer
    qa_pred = question + " " + bot_answer

    wv_qa = s2v_model.seq_vec_sent(qa)
    wv_qa_pred = s2v_model.seq_vec_sent(qa_pred)

    sims = {'cosine'    : cosine_sim(wv_qa, wv_qa_pred)[0][0],
            'euclidian' : euclidian_sim(wv_qa, wv_qa_pred)[0][0]}
    return sims


def evaluate_model_train_test(train_data, test_data, model_scheme):  
    """
    Evaluate model with cosine proximity as the train and test data are disjoint
    """
    questions = test_data["Question"].to_numpy()
    answers = test_data["Answer"].to_numpy()
    s2v_model = sent2vec(train_data, model_scheme)
    bot_answers = [answer_question(q, s2v_model, train_data, cosine_sim)[0] for q in questions]
    evaluations = [evaluate_answer_w2v(questions[i], answers[i], bot_answers[i][0], s2v_model)['cosine'] for i in range(len(questions))]
    return np.mean(evaluations)


def evaluate_model(data, s2v_model, k):
    """
    Checks if correct answer is in k answers provided by the model
    """
    questions = data["Question"].to_numpy()
    answers = data["Answer"].to_numpy()
    bot_answers = [answer_question(q, s2v_model, data, cosine_sim, k)[0] for q in questions]
    evaluations = [answers[i] in bot_answers[i] for i in range(len(questions))]
    return np.mean(evaluations)


def split_data(data, split=0.8):
    """
    Split data into train and test dataframes
    """
    N = len(data)
    N_train = int(N*split)
    indexes = list(range(N))
    np.random.shuffle(indexes)
    train_data = data.iloc[indexes[:N_train], :].reset_index(drop=True)
    test_data  = data.iloc[indexes[N_train:], :].reset_index(drop=True)
    return train_data, test_data



if __name__ == "__main__":
    
    #### Read data
    path = "../Data/youssef_data.csv"
    data = pd.read_csv(path, encoding="latin-1", header=None, names=["Question","Answer"]) 
    data["Question"] = data["Question"].apply(normalizeString)
    data["Answer"] = data["Answer"].apply(normalizeString) 


    #### Model evaluation : when the test and train sets are equal
    s2v = sent2vec(data, 'ws')
    print("#### Model precision : ", evaluate_model(data, s2v, 4), "\n")

    #### plot according per k
    accuracy_per_k = [evaluate_model(data, s2v, k) for k in range(1, 10)]
    plt.plot(range(1, 10), accuracy_per_k)
    plt.title("Model precision per number of answers considered")
    plt.show()

    #### Test evaluation: When the test and train sets are different
    index = np.random.randint(0, len(data))
    question = data.iloc[index, 0] 
    answer = data.iloc[index, 1]
    bot_answers, sims = answer_question(question, s2v, data, cosine_sim, 1)
    
    #### Print answer and evaluation
    print("#### Answer evaluation : sim(qa_pairs)\n")
    print("Question :", question, "\nAnswer :\n")
    for i, rep in enumerate(bot_answers):
        print("\t >>> Similarity : {} - {}\n".format(float(sims[i]), bot_answers[0]))

    print("Evaluations : ", evaluate_answer_w2v(question, answer, bot_answers[0], s2v))
    print("Evaluations : ", evaluate_answer(question, answer, bot_answers[0]))


    #### Evaluate model by mean of cosine similarity between GT and predicted question answer pairs
    print("\n#### average similarity of answers to evaluate model when train and test data are disjoint")
    train_data, test_data = split_data(data)
    print("Model precision : ", evaluate_model_train_test(train_data, test_data, 'ws'))
