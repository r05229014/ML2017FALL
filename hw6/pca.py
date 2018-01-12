from skimage import io
import skimage
import numpy as np
import sys
import os

path1 = sys.argv[1]
path2 = sys.argv[2]

def reconstruction(img):
	img = io.imread(img)
	img = img.flatten()
	y = img - X_mean
	y = np.dot(y,U)
	uuu = np.zeros(600*600*3)
    
	for i in range(4):
		uuu += y[i] * U[:,i]
	new = (uuu + X_mean)
	new -= np.min(new)
	new /= np.max(new)
	new = (new*255).astype(np.uint8)
	new = new.reshape(600,600,3)
	return new

img_data = []
for i in range(415):
    img_data.append(path1 + '/%s.jpg' %i)
    
X = np.zeros((1080000, 415))
k = 0
for i in img_data:
    #print(i)
    img = io.imread(i)
    img = img.flatten()
    X[:,k] = img
    k += 1

X_mean = np.mean(X, axis=1)

x = np.zeros(X.shape)
for i in range(X.shape[1]):
    x[:,i] = X[:,i] - X_mean
U, s, V = np.linalg.svd(x, full_matrices=False)
print(os.path.join(path1,path2))
ff = reconstruction(os.path.join(path1,path2))
io.imsave('reconstruction.jpg', ff)
