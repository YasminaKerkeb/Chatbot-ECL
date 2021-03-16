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
from preprocess import variablesFromPair
from utils import showPlot
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


def train(model, optimizer, train_iter, metadata, grad_clip, n_iters):
    model.train()  # put models in train mode (this is important because of dropout)

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    training_pairs = [variablesFromPair(random.choice(train_pairs),train_input_lang,train_output_lang)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
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

    showPlot(plot_losses)

    return total_loss / len(train_iter)


def main():
    #Load Data


    
    cuda = torch.cuda.is_available() 
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    device = torch.device('cuda' if cuda else 'cpu')

    print("Using %s for training" % ('GPU' if cuda else 'CPU'))
    print('Loading dataset...', end='', flush=True)    

    hidden_size = 100

    model=train_model_factory(input_size,hidden_size,output_size,n_layers,dropout_p)
    
  
    if cuda:
        model = nn.DataParallel(model, dim=1)  # if we were using batch_first we'd have to use dim=0
    print(model)  # print models summary

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    try:
        best_val_loss = None
        for epoch in range(args.max_epochs):
            start = datetime.now()
            # calculate train and val loss
            train_loss = train(model, optimizer, train_iter, metadata, args.gradient_clip)
            val_loss = evaluate(model, val_iter, metadata)
            print("[Epoch=%d/%d] train_loss %f - val_loss %f time=%s " %
                  (epoch + 1, args.max_epochs, train_loss, val_loss, datetime.now() - start), end='')

            # save models if models achieved best val loss (or save every epoch is selected)
            if args.save_every_epoch or not best_val_loss or val_loss < best_val_loss:
                print('(Saving model...', end='')
                save_model(args.save_path, model, epoch + 1, train_loss, val_loss)
                print('Done)', end='')
                best_val_loss = val_loss
            print()
    except (KeyboardInterrupt, BrokenPipeError):
        print('[Ctrl-C] Training stopped.')

    test_loss = evaluate(model, test_iter, metadata)
    print("Test loss %f" % test_loss)


if __name__ == '__main__':
    main()
