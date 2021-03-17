MAX_LENGTH = 15
stopwords=[]
SOS_token = 0
EOS_token = 1

teacher_forcing_ratio = 0.5
PAD_token= '<pad>'  # pad token
LSTM = 'LSTM'
GRU = 'GRU'
MODEL_FORMAT = "seq2seq-%s.pt"
MODEL_START_FORMAT = "seq2seq-%d"

DATA_PATH='data/youssef_data.csv'
TEST_SIZE=0.1
PARAMS={"hidden_size": 100,
        "n_layers":3,
        "dropout_p":0.1,
        "n_iters":5000,
        "num_epochs":2
        }

