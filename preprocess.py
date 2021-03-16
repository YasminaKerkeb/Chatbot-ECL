from config import *
import unicodedata
import re
from torch.autograd import Variable
import torch



class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split():
            if word == '':
                print('****************',sentence)
            self.addWord(word)
         
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z0-9?&\%\-]+", r" ", s)
    return s


def TrimWordsSentence(sentence):
    resultwords = [word for word in sentence.split() if word.lower() not in stopwords]
    resultwords = ' '.join(resultwords)
    return resultwords

def TrimWords(pairs):
    for pair in pairs: 
        pair[0] = TrimWordsSentence(pair[0])
        #pair[1] = pair[1].strip()
    return pairs

def filterPair(p):
    return len(p[0].split()) < MAX_LENGTH and \
        len(p[1].split()) < MAX_LENGTH 

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(data,questions, answers, reverse=False):
    input_lang, output_lang, pairs = readLangs(data,questions, answers, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = TrimWords(pairs)
    
    for pair in [pair for pair in pairs if not filterPair(pair)]:
        print('%s (%d) -> %s (%d)' % (pair[0],len(pair[0].split()),pair[1],len(pair[1].split())))  
    
    pairs = filterPairs(pairs)
    
    print('')
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        if '' in output_lang.word2index: print(pair[1].split())
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def readLangs(data,questions, answers, reverse=False):
    print("Reading lines...")
        
    # Split every line into pairs and normalize
    pairs = list(data.values)

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(answers)
        output_lang = Lang(questions)
    else:
        input_lang = Lang(questions)
        output_lang = Lang(answers)

    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    #return [lang.word2index[word] for word in sentence.split()]
    indexes=[]
    for word in sentence.split(' '):
        try:
            indexes.append(lang.word2index[word] )
        except KeyError:
                pass
    return indexes

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    return result

def variablesFromPair(pair,input_lang,output_lang):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)