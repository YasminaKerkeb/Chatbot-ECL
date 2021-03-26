import torch
import torch.nn as nn
import random
import string
from config import SOS_token, EOS_token, MAX_LENGTH
from torch.autograd import Variable
from src.preprocess import variableFromSentence
from .sampling import GreedySampler, RandomSampler, BeamSearch
from retrieve import answer_question
from models.retrieve.similarities import cosine_sim, euclidian_sim
from models.retrieve.sent2vec import sent2vec


class MultiSeq2SeqTrain(nn.Module):
    def __init__(self, encoder, decoder, teacher_forcing_ratio=0.5):
        """
        Encapsulates Seq2Seq model. This model is used for training seq2seq model, it returns (unscaled)
        probabilities for output which is needed for model training.

        :param encoder: Encoder.
        :param decoder: Decoder.
        :param teacher_forcing_ratio: Teacher forcing ratio. Default: 0.5.

        Inputs: question, trg
            - **question** (seq_len + 2, batch): Question sequence. It is expected that sequences have <SOS> token at start
            and <EOS> token at the end. +2 because question contains two extra tokens <SOS> and <EOS>.
            - **answer** (seq_len + 2, batch): Answer sequence. It is expected that sequences have <SOS> token at start and
            <EOS> token at the end. +2 because answer contains two extra tokens <SOS> and <EOS>.

        Outputs: outputs
            - **outputs** (seq_len + 1, batch, vocab_size): Model predictions for output sequence. These are raw
            unscaled logits. First dimension is (seq_len + 1) because we return predictions for next word for all tokens
            except last one (<EOS>), which means seq2seq will feed in decoder following sequence [<SOS>, tok1, tok2, ...
            , tokN] (notice no <EOS> at the end) and return next word prediction for each one of them.
        """
        super(MultiSeq2SeqTrain, self).__init__()
        self.encoders = [encoder]*4
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio


    

    def forward(self, question, answer, retrieve_candidates,criterion = nn.NLLLoss()):
    
        target_length = answer.size()[0]
        encoder_hidden = self.encoders[0].initHidden()
        encoded_outputs=self.full_encode(question,self.encoders[0])
        n=len(retrieve_candidates)
        for i in range(len(retrieve_candidates)):
   
            reply=retrieve_candidates[i]
            encoded_reply=self.full_encode(reply,self.encoders[i+1])
            encoded_outputs=encoded_outputs+encoded_reply
        
        
        encoder_outputs=encoded_outputs
       

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        loss=0
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            
                
                loss += criterion(decoder_output, answer[di])
                decoder_input = answer[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                
                
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0].item()

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input
                loss += criterion(decoder_output, answer[di])

                if ni == EOS_token:
                    break

        loss.backward()
        



        return loss.item() / target_length


    def full_encode(self,input_text,encoder):
        
        encoder_hidden = encoder.initHidden()
        input_length = input_text.size()[0]

        encoder_outputs = Variable(torch.zeros(MAX_LENGTH, encoder.hidden_size))
        

        for ei in range(input_length):
            
            encoder_output, encoder_hidden = encoder(
                input_text[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        return encoder_outputs

        


    


class MultiSeq2SeqPredict(nn.Module):
    """
    This class is wrapper around pre-trained model which can be used for testing model.
    This model takes (numericalized) input, delegates it to appropriate sequence sampler and returns

    :param encoder: Pre-trained encoder.
    :param decoder: Pre-trained decoder.
    :param field: Torchtext Field object which handles processing of raw data into tensors.

    Inputs: questions, sampling_strategy, max_seq_len
        - **questions** list(str): List of raw question strings.
        - **sampling_strategy** (str): Strategy for sampling output sequences. ['greedy', 'random', 'beam_search']
        - **max_seq_len** (scalar): Maximum length of output sequence.

    Outputs: sequences
        - **sequences** list(str): List of answers sequences generated by model.
    """
    def __init__(self, pretrained_model):
        super(MultiSeq2SeqPredict, self).__init__()
        self.encoders = pretrained_model.encoder
        self.decoder = pretrained_model.decoder
      
    def forward(self,question, retrieve_candidates,train_input_lang,train_output_lang):
        
        
        input_variable = variableFromSentence(train_input_lang, question)
        
        input_length = input_variable.size()[0]

        encoder_hidden = self.encoders[0].initHidden()
        encoded_outputs=self.full_encode(question,self.encoders[0])
        n=len(retrieve_candidates)
        for i in range(len(retrieve_candidates)):
   
            reply=retrieve_candidates[i]
            encoded_reply=self.full_encode(reply,self.encoders[i+1])
            encoded_outputs=encoded_outputs+1/n*encoded_reply
        
        
        encoder_outputs=encoded_outputs

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
        decoder_input = decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)

        for di in range(MAX_LENGTH):
            
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data

            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0].item()

            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(train_output_lang.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))

        return decoded_words, decoder_attentions[:di + 1]



    def test(self, question,answer,criterion = nn.NLLLoss()):
        # raw strings to tensor

        input_length = question.size()[0]
        target_length = answer.size()[0]
        encoder_hidden = self.encoder.initHidden()


        encoder_outputs = Variable(torch.zeros(MAX_LENGTH, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(question[ei],
                                                    encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
        decoder_input = decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(MAX_LENGTH,MAX_LENGTH)
        loss=0

        for di in range(target_length):

            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #loss += criterion(decoder_output, target_variable[di])
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0].item()

            loss += criterion(decoder_output, answer[di])
            if ni == EOS_token:
                break


        return loss.item() / target_length


