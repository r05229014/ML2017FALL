import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten

f = open('./data/train.csv','r')
datas = f.readlines()[1:]
Y_train = []
X_train = []
n = 0
for data in datas:
    X_train.append([])
    label, pixels = data.split(',')
    pixels = np.array(list(map(float,pixels.split(' '))))
    X_train[n] = pixels.reshape((48,48,1))
    Y_train.append(float(label))
    n += 1

X_train = np.array(X_train) / 255
Y_train = np.array(Y_train)

## Initialize
model =  Sequential()
## add CNN 
model.add(Conv2D(25, (3, 3), input_shape=(48,48,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(50, (3, 3)))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.2))
model.add(Flatten())
## add DNN
model.add(Dense(120, kernel_initializer="uniform", use_bias=True,activation = 'relu'))
model.add(Dropout(0.7))
model.add(Dense(7, kernel_initializer="uniform", use_bias=True,activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
### fit
model.fit(X_train, Y_train, batch_size=100, epochs=50)

from keras.models import load_model
model.save('my_model.h5')
