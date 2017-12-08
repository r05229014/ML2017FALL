import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dense, Dropout
from keras.layers import LSTM, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import logging
import pickle 


def tokenizer_data_toidx(line_data):
    index_data = []
    i = 0
    for line in line_data:
        index_data.append([])    
        for word in line.split():
            if word in word_tr.wv:
                index_data[i].append(word_tr.wv.vocab[word].index)
        i += 1  
    return index_data

def RNN(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 128,
                            weights=[embedding_matrix],
                            input_length=36,
                            trainable=False))
    model.add(LSTM(300, return_sequences=True, dropout=0.3)) 
    model.add(LSTM(300, return_sequences=False, dropout=0.3)) 
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
               metrics=['accuracy'])
    
    return model



#paths
train_label_path = sys.argv[1]
train_nolabel_path = sys.argv[2]


# model parameter
batch_size = 128
epochs = 50
max_words = 20000
max_padding_len = 36
embedding_dim = 32
hidden_size = 100
dropout_rate = 0.3
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#read data
train = pd.read_fwf(train_label_path, header=None, encoding='utf_8',widths=[1, 8, 200])
#test = pd.read_fwf('./data/testing_data.txt', encoding='utf_8', header=None, skiprows=1)
train_nolabel = pd.read_fwf(train_nolabel_path, encoding='utf_8', header=None, widths=[200])
y = np.asarray(train[0])

#filter
train = train[2].apply(lambda x:''.join([i for i in x if i in 'abcdefghijklmnopqrstuvwxyz ']))
#test = test[0].apply(lambda x:''.join([i for i in x if i in 'abcdefghijklmnopqrstuvwxyz ']))
train_nolabel = train_nolabel[0].apply(lambda x:''.join([i for i in x if i in 'abcdefghijklmnopqrstuvwxyz ']))
alltext = np.append(train, train_nolabel)


new = []
for i in alltext:
    new.append(i.split())
word_tr = Word2Vec.load('./size128_THISSSSSSSSSSS.model.bin')
vocab_size = len(word_tr.wv.vocab)

index_x = tokenizer_data_toidx(train)
X = pad_sequences(index_x, max_padding_len)
embedding_matrix = np.zeros((len(word_tr.wv.vocab), 128))
for i in range(len(word_tr.wv.vocab)):
    embedding_vector = word_tr.wv[word_tr.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

        
call = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')


model = RNN(vocab_size)
model.fit(X, y, validation_split=0.1, batch_size=64, epochs=200, shuffle=True, callbacks=[call])

model = model.save('./model_trained_new.h5')



