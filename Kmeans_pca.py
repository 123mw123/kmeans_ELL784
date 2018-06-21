import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = pd.read_csv("2015EE10466.csv")
data1 = pd.read_csv("2015EE10466.csv")
print(data.describe())
labels = list(data['pixel784'])
data = data.iloc[0:,:784]
data1 = data1.iloc[0:, :784]
data1 = np.array(data1)
pca = PCA()
pca.fit(data)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(len(pca.explained_variance_))
print(len(pca.explained_variance_ratio_))
q=float(0)
j =0

p=pca.explained_variance_ratio_
for i in p:
    if(q<0.90):
        q = q+i
        j = np.where(p == i)


print(q)
print(j)

res_var = [1]
for i in range(1,785):
    res_var = res_var + [1.0 -sum(p[:i])]
plt.plot(range(785),res_var)
plt.show()

pca = PCA(n_components=85)
data = pca.fit_transform(data)

clf = KMeans(n_clusters=10, n_init=10, max_iter=100)
clf.fit(data)
pred = clf.predict(data)
print(pred)
cluster_dict ={}

for i,val in enumerate(pred):
    if(val not in cluster_dict.keys()):
        cluster_dict[val]= []
    cluster_dict[val] = cluster_dict[val] + [i]

print(cluster_dict.keys())

sum1 = 0
for j in cluster_dict.keys():

    lis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in cluster_dict[j]:
        lis[labels[i]] = lis[labels[i]]+1
    print('cluster:', lis.index(max(lis)))
    print(lis)
    print(sum(lis),'sum')
    print(max(lis))
    sum1 = sum1 + sum(lis) -max(lis)

    print(sum1)
    print(sum(lis)-max(lis))

print(sum1/3000,'error')

print(adjusted_rand_score(labels,pred))



j =0
fig, ax = plt.subplots()
colors = cm.rainbow(np.linspace(0, 1, 10))
for i in range(10):
    ax.scatter(data[j:j+300,0],data[j:j+300,1],label=i,color=colors[i])
    j=j+300
ax.legend()
ax.grid(True)
plt.show()

print(pca.components_)
print(pca.components_.shape)

comp = [784,1,2,3,4,5,20,30,40,50,60,70,85,100]
for i in comp:
    pca = PCA(n_components=i)
    dim_red = pca.fit_transform(data1)
    x = pca.inverse_transform(dim_red[901])
    print(x.shape)
    x = np.reshape(x,(28,28))
    plt.imshow(x,cmap='Blues')
    plt.title(i)
    plt.show()