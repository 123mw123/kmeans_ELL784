from scipy import *
from pylab import *
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("2015EE10466.csv")
labels = list(data['pixel784'])
data = data.iloc[0:,:784]

data = np.array(data)
#data[data>0] = 1
img = np.reshape(data[901,:784],(28,28))
img = np.array(img)
print(img.shape)
plt.imshow(img,cmap='Blues')
plt.show()
m,n = img.shape
U,S,Vt = svd(img)
S = resize(S,[m,1])*eye(m,n)
print(S)


for k in range(0,5):
    plt.imshow(dot(U[:,k:k+1],dot(S[k:k+1,:],Vt[:,:])),cmap='Blues')
    plt.title(k+1)
    plt.show()
k = 85
plt.imshow(dot(U[:,0:k],dot(S[0:k,:],Vt[:,:])),cmap='Blues')
plt.show()