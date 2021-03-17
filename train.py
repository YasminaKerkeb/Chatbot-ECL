import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from model import train_model_factory
from serialization import save_object, save_model, save_vocab
from datetime import datetime
from models.seq2seq.model import Seq2SeqTrain, Seq2SeqPredict
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pandas as pd
import time
from preprocess import variablesFromPair, prepareData, normalizeString
from utils import showPlot, timeSince
import random






def evaluate(model, val_iter, metadata):
    model.eval()  # put models in eval mode (this is important because of dropout)

    total_loss = 0
    with torch.no_grad():
        for batch in val_iter:
            # calculate models predictions
            question, answer = batch.question, batch.answer
            logits = model(question, answer)

            # calculate batch loss
            loss = F.cross_entropy(logits.view(-1, metadata.vocab_size), answer[1:].view(-1),
                                   ignore_index=metadata.padding_idx)  # answer[1:] skip <sos> token
            total_loss += loss.item()

    return total_loss / len(val_iter)


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
    return plot_loss_avg

    


def main():
    hidden_size = 300
    n_layers=1
    dropout_p=0.1
    n_iters=5000
    num_epochs=1

    cuda = torch.cuda.is_available() 
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    device = torch.device('cuda' if cuda else 'cpu')

    print("Using %s for training" % ('GPU' if cuda else 'CPU'))
    print('Loading dataset...', end='', flush=True)  

    #Loading data
    data=pd.read_csv('data/youssef_data.csv',encoding="latin-1",header=None,names=["Question","Answer"]) 
    data["Question"]=data["Question"].apply(normalizeString)
    data["Answer"]=data["Answer"].apply(normalizeString) 
    train_input_lang, train_output_lang, train_pairs = prepareData(data,'questions', 'answers', False)
    
    input_size=train_input_lang.n_words
    output_size=train_output_lang.n_words
    training_pairs = [variablesFromPair(random.choice(train_pairs),train_input_lang,train_output_lang)
                      for i in range(n_iters)]
    model=train_model_factory(input_size,hidden_size,output_size,n_layers,dropout_p)
    
  
    if cuda:
        model = nn.DataParallel(model, dim=1)  # if we were using batch_first we'd have to use dim=0
    print(model)  # print models summary

    

    try:
        best_val_loss = None
        for epoch in range(num_epochs):
            start = datetime.now()
            # calculate train and val loss
            train_loss = train(model, training_pairs,  n_iters)
            #val_loss = evaluate(mode)
            #print("[Epoch=%d/%d] train_loss %f - val_loss %f time=%s " %
            #      (epoch + 1, num_epochs, train_loss, val_loss, datetime.now() - start), end='')

            # save models if models achieved best val loss (or save every epoch is selected)
            # if  not best_val_loss or val_loss < best_val_loss:
            #     print('(Saving model...', end='')
            #     #save_model(args.save_path, model, epoch + 1, train_loss, val_loss)
            #     print('Done)', end='')
            #     best_val_loss = val_loss
            # print()
    except (KeyboardInterrupt, BrokenPipeError):
        print('[Ctrl-C] Training stopped.')

    #test_loss = evaluate(model, test_iter, metadata)
    #print("Test loss %f" % test_loss)


if __name__ == '__main__':
    main()
