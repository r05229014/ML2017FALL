from gensim.models import Word2Vec
import sys
import io
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle 
import csv

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

def predict(X_padding, csv_name):
    model = load_model('./weights-improvement-08-0.82.hdf5')
    print(model.summary())
    y_predict = model.predict(X_padding, batch_size=256)
    ans = [round(x[0]) for x in y_predict]
    out = []
    out.append(["id", "label"])
    for i in range(len(ans)):
        out.append([i,int(ans[i])])
        
    with open(csv_name, 'w+') as f:
        s = csv.writer(f, delimiter=',', lineterminator='\n')
        for i in range(len(out)):
            s.writerow(out[i])

max_seq_len = 36
embedding_dim = 128
test_path = sys.argv[1]
out_path = sys.argv[2]

# data_path and preprocessing
test_data = pd.read_fwf(test_path, encoding='utf_8', header=None, skiprows=1)
test_data = test_data[0].apply(lambda x:''.join([i for i in x if i in 'abcdefghijklmnopqrstuvwxyz ']))
word_tr = Word2Vec.load('./size128_THISSSSSSSSSSS.model.bin') #load tokenizer
vocab_size = len(word_tr.wv.vocab)
index_test = tokenizer_data_toidx(test_data)
pad_text = pad_sequences(index_test, max_seq_len)
predict(pad_text, out_path)
