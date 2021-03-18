from model import train_model_factory, val_model_factory, predict_model_factory
from src.preprocess import normalizeString, prepareData, TrimWordsSentence
from sklearn.model_selection import train_test_split
from config import DATA_PATH, TEST_SIZE
import pandas as pd
import re
import torch


class ChatBot(object):

    def __init__(self,model_path):
        self.prepare_data()
        input_size=self.train_input_lang.n_words
        output_size=self.train_output_lang.n_words
        self.model=predict_model_factory(model_path,input_size,output_size)
        self.model.eval()
        
        
    def reply(self, input_text):
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
        self.train_input_lang, self.train_output_lang,_ = prepareData(train_data,'questions', 'answers', False)
        