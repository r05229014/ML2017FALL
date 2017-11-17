import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
import sys

x_train_path = sys.argv[1]

### input train.csv
f = open(x_train_path,'r')
train_csv = f.readlines()[1:]
y = []
X = []
n = 0
for data in train_csv:
    X.append([])
    label, pixels = data.split(',')
    pixels = np.array(list(map(float,pixels.split(' '))))
    X[n] = pixels
    y.append(float(label))
    n += 1
f.close()
X = np.array(X)/255
y = np.array(y)
y = np.eye(7)[list(map(int,y))]


### feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X = X.reshape(len(X),48,48,1)

### model structure
def CNN(lr):
    lr = lr
    
    model =  Sequential()
    
    # add conv1 layers
    model.add(Conv2D(32, (5, 5), use_bias=True, padding='SAME',strides=1, activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    # add conv2 layers
    model.add(Conv2D(64, (3, 3), use_bias=True,padding='SAME',strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    # add conv3 layers
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), use_bias=True,padding='SAME',strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    # add conv3 layers
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), use_bias=True,padding='SAME',strides=1, activation='relu'))
    model.add(Dropout(0.2))
    
    # add conv4 layers
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), use_bias=True,padding='SAME',strides=1, activation='relu'))
    model.add(Dropout(0.2))
    
    # add conv5 layers
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), use_bias=True,padding='SAME',strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # add conv6 layers
    #model.add(BatchNormalization())
    #model.add(Conv2D(512, (3, 3), use_bias=True,padding='SAME',strides=1, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))


    # add DNN1
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128, kernel_initializer="random_uniform", use_bias='Ones',activation = 'elu'))
    model.add(Dropout(0.6))
    
    # add DNN2
    model.add(BatchNormalization())
    model.add(Dense(256, kernel_initializer="random_uniform", use_bias='Ones',activation = 'elu'))
    model.add(Dropout(0.6))
    
    # add DNN3
    model.add(BatchNormalization())
    model.add(Dense(512, kernel_initializer="random_uniform", use_bias='Ones',activation = 'elu'))
    model.add(Dropout(0.5))
    
    # add DNN4
    model.add(BatchNormalization())
    model.add(Dense(1024, kernel_initializer="random_uniform", use_bias='Ones',activation = 'elu'))
    model.add(Dropout(0.5))
    
    # add classifier
    model.add(Dense(7, kernel_initializer="random_uniform", use_bias=True,activation = 'softmax'))

    # optimizer
    opt = Adam(lr=lr)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

model = CNN(0.001)
model.fit(X, y, validation_split=0.2, batch_size=256, epochs=100, shuffle=True)

model.save('./mymodel.hdf5')