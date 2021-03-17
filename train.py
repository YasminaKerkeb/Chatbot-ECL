import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from model import train_model_factory, val_model_factory
from src.serialization import save_object, save_model, save_vocab, save_metrics
from datetime import datetime
from models.seq2seq.model import Seq2SeqTrain, Seq2SeqPredict
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from src.preprocess import normalizeString, prepareData, variablesFromPair, variableFromSentence
from src.utils import showPlot, timeSince
import random
from config import DATA_PATH, TEST_SIZE, PARAMS



#Parameters





def evaluate(model, test_pairs, test_input_lang):
    model.eval()  # put models in eval mode (this is important because of dropout)
    print_loss_total=0
    with torch.no_grad():
        for iter in range(len(test_pairs)):
            test_pair = test_pairs[iter]
            input_variable = test_pair[0]
            target_variable = test_pair[1]
            input_variable = variableFromSentence(test_input_lang, input_variable)
            target_variable = variableFromSentence(test_input_lang, target_variable)
            loss=model.test(input_variable, target_variable)
            
            print_loss_total += loss
        
    return print_loss_total/len(test_pairs)
    


def train(model,training_pairs, n_iters, print_every=1000,plot_every=100):
    model.train()  # put models in train mode (this is important because of dropout)
    encoder=model.encoder
    decoder=model.decoder
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(model.encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.SGD(model.decoder.parameters(), lr=0.01)
    
    
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = model(input_variable, target_variable)

        encoder_optimizer.step()
        decoder_optimizer.step()
        
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses,[encoder.n_layers,encoder.hidden_size])
    
    return print_loss_avg

    


def main():
    #Main

    cuda = torch.cuda.is_available() 
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    device = torch.device('cuda' if cuda else 'cpu')

    print("Using %s for training" % ('GPU' if cuda else 'CPU'))
    print('Loading dataset...', end='', flush=True)  

    #Loading data
    data=pd.read_csv(DATA_PATH ,encoding="latin-1",header=None,names=["Question","Answer"]) 
    data["Question"]=data["Question"].apply(normalizeString)
    data["Answer"]=data["Answer"].apply(normalizeString) 

    #Split into train, test set
    train_data, test_data = train_test_split(data, test_size=TEST_SIZE,random_state=11)
    train_input_lang, train_output_lang, train_pairs = prepareData(train_data,'questions', 'answers', False)
    test_input_lang, test_output_lang, test_pairs = prepareData(test_data,'questions', 'answers', False)
    
    input_size=train_input_lang.n_words
    output_size=train_output_lang.n_words
    training_pairs = [variablesFromPair(random.choice(train_pairs),train_input_lang,train_output_lang)
                      for i in range(PARAMS["n_iters"])]
    model=train_model_factory(input_size,PARAMS["hidden_size"],output_size,PARAMS["n_layers"],PARAMS["dropout_p"])
    
    params=PARAMS.copy()
    if cuda:
        model = nn.DataParallel(model, dim=1)  # if we were using batch_first we'd have to use dim=0
    print(model)  # print models summary


    try:
        best_train_loss = 1e3
        for epoch in range(PARAMS["num_epochs"]):
            start = datetime.now()
            # calculate train and val loss
            train_loss = train(model, training_pairs, PARAMS["n_iters"])
            #val_loss = evaluate(mode)
            print("\n\n[Epoch=%d/%d] train_loss %f time=%s " %
                  (epoch + 1, PARAMS["num_epochs"], train_loss,datetime.now() - start), end='')

            # save models if models achieved best val loss (or save every epoch is selected)
            if  train_loss < best_train_loss:
                params['train_loss']=train_loss
                best_train_loss = train_loss
                best_model=model

            # print()
    except (KeyboardInterrupt, BrokenPipeError):
        print('[Ctrl-C] Training stopped.')
    
    trained_model=val_model_factory(best_model)
    test_loss = evaluate(trained_model, test_pairs, test_input_lang)
    print('\n\nSaving model...', end='')
    now=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_model('best_models/', best_model, now)
    print('\n\nDone', end='')
    print("\n\nTest loss %f" % test_loss)
    print('\n\nSaving metrics...', end='')
    params["test_loss"]=test_loss
    save_metrics({**{"datetime":now},**params})
    print('\n\nDone\n\n', end='')



if __name__ == '__main__':
    main()
