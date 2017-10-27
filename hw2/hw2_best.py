import numpy as np
import csv
import sys

test_path = sys.argv[1]
out_path = sys.argv[2]


X = []
with open('./data/X_train') as f:
    row = csv.reader(f, delimiter =",")
    next(row,None)
    for r in row:
        X.append(list(map(float,r)))
        
X = np.array(X)
'''
y = []
with open('./data/Y_train') as f:
    row = csv.reader(f, delimiter =",")
    next(row,None)
    for r in row:
        y.append(list(map(float,r)))
        
y = np.array(y)
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping

classfier = Sequential()

#hidden1
classfier.add(Dense(106, kernel_initializer="uniform", use_bias=True, input_shape=(106,)))
classfier.add(BatchNormalization())
classfier.add(Activation('relu'))
classfier.add(Dropout(0.5))

classfier.add(Dense(1, kernel_initializer="uniform", use_bias=True, activation="sigmoid"))

classfier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#call = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 30)
#classfier.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=50, nb_epoch=200, callbacks=[call])
classfier.fit(X, y, batch_size=300, nb_epoch=150)
classfier.save('my_model.h5')
'''
from keras.models import load_model
classfier = load_model('my_model.h5')

x_test = []
with open(test_path) as f:
    row = csv.reader(f, delimiter =",")
    next(row,None)
    for r in row:
        x_test.append(list(map(float,r)))
        
x_test = np.array(x_test)
x_test = sc.fit_transform(x_test)

pre = classfier.predict(x_test)
pre = pre > 0.5

ans = []
ans.append(["id", "label"])
for i in range(pre.shape[0]):
    ans.append([i+1,int(pre[i])])

filename = out_path
with open(filename,'w+') as f:
    s = csv.writer(f,delimiter=',',lineterminator='\n')
    for i in range(len(ans)):
        s.writerow(ans[i])

