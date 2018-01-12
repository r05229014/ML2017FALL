from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from keras.models import load_model
import sys

path1 = sys.argv[1]
path2 = sys.argv[2]
path3 = sys.argv[3]



#train_num = 130000
X = np.load(path1)
X = X.astype('float32') / 255.
X = np.reshape(X, (len(X), -1))
#x_train = X[:train_num]
#x_val = X[train_num:]
#x_train.shape, x_val.shape

#input_img = Input(shape=(784,))

#encoded = Dense(128, activation='selu')(input_img)
#encoded = Dense(64, activation='selu')(encoded)
#encoded = Dense(32, activation='selu')(encoded)

#decoded = Dense(64, activation='selu')(encoded)
#decoded = Dense(128, activation='selu')(decoded)
#decoded = Dense(784, activation='selu')(decoded)

# build encoder
#encoder = Model(input=input_img, output=encoded)

# build autoencoder 
#adam = Adam(lr=5e-4)
#autoencoder = Model(input=input_img, output=decoded)
#autoencoder.compile(optimizer='adam', loss='mse')
#autoencoder.summary()
#earlystopping = EarlyStopping(monitor='val_loss', patience = 5, verbose=1, mode='min')
#history = autoencoder.fit(x_train, x_train,
#                epochs=100,
#                batch_size=256,
#                shuffle=True,
#                validation_data=(x_val,x_val),
#                callbacks=[earlystopping])
#autoencoder.save('autoencoder2.h5')
#encoder.save('encoder2.h5')
encoder = load_model('encoder2.h5')

encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)

# get test cases
f = pd.read_csv(path2)
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])

# predict
o = open(path3,'w')
o.write("ID,ANS\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1 # two images are in the same cluster
    else:
        pred = 0 # two images are not in the same cluster
    o.write("{},{}\n".format(idx,pred))
o.close()
