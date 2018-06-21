import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import numpy as np

data = pd.read_csv("2015EE10466.csv")

print(data.describe())
labels = list(data['pixel784'])
data = data.iloc[0:,:784]



clf = KMeans(n_clusters=5, n_init=10, max_iter=100)
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
err = []
for j in cluster_dict.keys():

    lis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in cluster_dict[j]:
        lis[labels[i]] = lis[labels[i]]+1
    print('cluster:', lis.index(max(lis)))
    print(lis)
    print(sum(lis),'sum')
    print(max(lis))
    sum1 = sum1 + sum(lis) -max(lis)
    err = err +[(max(lis)/sum(lis))]
    print(sum1)
    print(sum(lis)-max(lis))

print(sum1/3000,'error')

print(adjusted_rand_score(labels,pred))

print((1 - (sum(err)/len(err))))

