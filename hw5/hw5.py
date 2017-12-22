import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dot, Add, Input, Merge, Embedding, Concatenate
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.models import load_model
from keras import backend as K
import sys


###########################
HIDDEN_SIZE = 256
BATCH_SIZE = 2000
EPOCHS = 200
LATENT_DIM = 1000
VALIDATION_SPLIT = 0.1
DROPOUT_RATE = 0.1
#filepath = './model_nn_laten=.h5'
outpath = sys.argv[2]
#train_path = './data/train.csv'
test_path = sys.argv[1]
###########################
def plothistory(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss(latent_dim = %s)' %LATENT_DIM)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.ylim((0,1))
    plt.savefig('train_nn_history%s' %LATENT_DIM)

def MF_model(n_users, n_items, latent_dim, drop_rate):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    
    user_vec = Embedding(n_users, latent_dim, input_length=1)(user_input)
    user_vec = Flatten()(user_vec)
    user_vec = Dropout(drop_rate)(user_vec)
        
    item_vec = Embedding(n_items, latent_dim, input_length=1)(item_input)
    item_vec = Flatten()(item_vec)
    item_vec = Dropout(drop_rate)(item_vec)
    
    user_bias = Embedding(n_users, 1, input_length=1)(user_input)
    user_bias = Flatten()(user_bias)
    user_bias = Dropout(drop_rate)(user_bias)
    
    item_bias = Embedding(n_items, 1, input_length=1)(item_input)
    item_bias = Flatten()(item_bias)
    item_bias = Dropout(drop_rate)(item_bias)
    
    r_hat = Dot(axes=1)([user_vec, item_vec])
    #r_hat = Add()([r_hat, user_bias, item_bias])
    model = Model(inputs=[user_input, item_input], outputs=r_hat)
    model.compile(loss='mse', optimizer='adam')
    return model

def NN_model(n_users, n_items, latent_dim, drop_rate, hidden_size):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    
    user_vec = Embedding(n_users, latent_dim, input_length=1)(user_input)
    user_vec = Flatten()(user_vec)
    user_vec = Dropout(drop_rate//2)(user_vec)
        
    item_vec = Embedding(n_items, latent_dim, input_length=1)(item_input)
    item_vec = Flatten()(item_vec)
    item_vec = Dropout(drop_rate//2)(item_vec)
    
    merge_vec = Concatenate()([user_vec, item_vec])
    hidden = Dense(hidden_size, activation='relu')(merge_vec)
    hidden = Dropout(drop_rate*2)(hidden)
    hidden = Dense(hidden_size//2, activation='relu')(hidden)
    hidden = Dropout(drop_rate*2)(hidden)
    hidden = Dense(hidden_size//4, activation='relu')(hidden)
    hidden = Dropout(drop_rate*2)(hidden)
    
    output = Dense(1)(hidden)
    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(loss='mse', optimizer='adam')
    return model
#################################################### main

#train_data = pd.read_csv(train_path, index_col=None)
test_data = pd.read_csv(test_path, index_col=None)
# TESTING
user_test = test_data['UserID'].values
movie_test = test_data['MovieID'].values

# data
#user = train_data['UserID']
#movie = train_data['MovieID']
#y = train_data['Rating']

#indices = np.arange(user.shape[0])
#np.random.shuffle(indices)
#user = user[indices]
#movie = movie[indices]
#y = y[indices]

# split data to train and val
#nb_validation_samples = int(VALIDATION_SPLIT * user.shape[0])
#user_train = user[nb_validation_samples:]
#user_val = user[0:nb_validation_samples]
#movie_train = movie[nb_validation_samples:]
#movie_val = movie[0:nb_validation_samples]
#y_train = y[nb_validation_samples:]
#y_val = y[0:nb_validation_samples]

# train
#n_users = max(user)
#n_items = max(movie)
#checkpointe = ModelCheckpoint(filepath=filepath,verbose=1,monitor='val_loss',save_best_only=True)
#earlystopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=1, mode='min')
#model = NN_model(n_users, n_items, LATENT_DIM, DROPOUT_RATE,HIDDEN_SIZE) ##########################################
#history = model.fit([user_train, movie_train], y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
#                    validation_data=([user_val, movie_val], y_val),
#                    verbose=1, callbacks=[checkpointe, earlystopping])
#plothistory(history)


model = load_model('./model_mf_laten=256.h5')
y_test = model.predict([user_test, movie_test], verbose=1)

f = open(outpath, 'w')
f.write("TestDataID,Rating\n")
for i in range (y_test.shape[0]):
    if (y_test[i][0] > 5):
        f.write(str(i+1) + ",5\n")
    elif (y_test[i][0] < 1):
        f.write(str(i+1) + ",1\n")
    else:
        f.write(str(i+1) + "," + str(y_test[i][0]) + "\n")
f.close()

