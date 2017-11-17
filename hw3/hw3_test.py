import numpy as np
import pickle
from keras.models import load_model
import csv
import sys

scalerfile = 'scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))


x_test_path = sys.argv[1]
out_path = sys.argv[2]

f = open(x_test_path,'r')
test_cev = f.readlines()[1:]
test_ = []
n=0
for data in test_cev:
    test_.append([])
    idd,pixels = data.split(',')
    pixels = np.array(list(map(float,pixels.split(' '))))
    test_[n] = pixels
    n += 1
f.close()
test_ = np.array(test_)/255

scalerfile = 'scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))
test_ = scaler.transform(test_)
test_ = test_.reshape(len(test_),48,48,1)

model = load_model('./weights-improvement-20-0.65.hdf5')

ans = model.predict(test_)
result = []
for i in range(ans.shape[0]):
    result.append(np.argmax(ans[i]))
result = np.array(result)

out = []
out.append(["id", "label"])
for i in range(result.shape[0]):
     out.append([i,result[i]])

filename = out_path
with open(out_path,'w+') as f:
    s = csv.writer(f,delimiter=',',lineterminator='\n')
    for i in range(len(out)):
        s.writerow(out[i])


