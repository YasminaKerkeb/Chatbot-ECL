import torch
from models.seq2seq.encoder import encoder_factory
from models.seq2seq.attention import decoder_factory
from models.seq2seq.model import Seq2SeqTrain, Seq2SeqPredict
from collections import OrderedDict



def train_model_factory(input_size,hidden_size,output_size,n_layers,dropout_p):
    encoder = encoder_factory(input_size,hidden_size,n_layers)
    decoder = decoder_factory(hidden_size, output_size,n_layers, dropout_p)
    return Seq2SeqTrain(encoder, decoder)

def val_model_factory(trained_model):
    return Seq2SeqPredict(trained_model)



def predict_model_factory(model_path):
    train_model = torch.load("best_models/"+model_path)
    return Seq2SeqPredict(train_model)
    

def get_state_dict(model_path):
    # load state dict and map it to current storage (CPU or GPU)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    return state_dict