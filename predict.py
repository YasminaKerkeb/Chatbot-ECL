from model import train_model_factory, val_model_factory, predict_model_factory
from src.preprocess import normalizeString, prepareData, TrimWordsSentence
from sklearn.model_selection import train_test_split
from config import DATA_PATH, TEST_SIZE
from models.retrieve.similarities import cosine_sim
import pandas as pd
from models.retrieve.sent2vec import sent2vec
import re
import json
import torch
from retrieve import answer_question
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

class ChatBot(object):

    def __init__(self,model_path,mode):
        self.mode=mode
        self.prepare_data()
        if self.mode=="retrieve":
            self.vectorize_data()
        else: 
            input_size=self.train_input_lang.n_words
            output_size=self.train_output_lang.n_words
            self.model=predict_model_factory(model_path,input_size,output_size)
            self.model.eval()

        
        
    def reply(self, input_text):
        if self.mode=="generate":
            with torch.no_grad():
                sentences = [s.strip() for s in re.split('[\.\,\?\!]' , input_text)]
                sentences = sentences[:-1]
                if sentences==[]:
                    sentences=[input_text]
                for sentence in sentences : 
                    trimmed_sentence= TrimWordsSentence(normalizeString(sentence))
                    print(trimmed_sentence)
                    answer_words, _ =self.model(trimmed_sentence,self.train_input_lang,self.train_output_lang)
                    answer = ' '.join(answer_words)
        
        elif self.mode=='retrieve':
            ### Answer question
            if input_text not in ['quit', 'q']: 
                
                _,responses = answer_question(input_text, self.s2v, self.w2v, self.data, cosine_sim, 1)
                answer=responses[0]
        
        else:
            answer="No mode specified"

            
        return answer
 

    def test_run(self,ques):
       answer=self.reply(ques)
       return answer

    
    def prepare_data(self):
        #Loading data
        data=pd.read_csv(DATA_PATH ,encoding="latin-1",header=None,names=["Question","Answer"]) 
        data["Question"]=data["Question"].apply(normalizeString)
        data["Answer"]=data["Answer"].apply(normalizeString) 
        #Split into train, test set
        train_data,_ = train_test_split(data, test_size=TEST_SIZE,random_state=11)
        self.data=data
        if self.mode=="generate":
            self.train_input_lang, self.train_output_lang,_ = prepareData(train_data,'questions', 'answers', False)
        
    def vectorize_data(self):
        ### Question
        self.s2v = sent2vec(self.data, 'cooc_2')
        self.w2v = self.s2v.get_model()

    def get_topics_data(self,input_text):
        with open('data/topics_data.json') as json_file:
            data = json.load(json_file)
        data=pd.DataFrame.from_dict(data, orient='index').reset_index()
        data.columns=["Topic","Answer"]
        data["similarity"]=data["Topic"].apply(lambda x:sentence_bleu([input_text],[x]))
        query=data.similarity == data.similarity.max()

        return data[query]["Topic"][0], data[query]["Answer"][0]

