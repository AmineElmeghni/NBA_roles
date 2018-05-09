import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from scipy import cluster

data = pd.read_csv("fullstats.csv")
#drop player name
data = data.drop("Player",1)
#turn blanks into zeroes
data= data.apply(pd.to_numeric, errors='coerce')

corr_data = pd.DataFrame.corr(data)

#normalize numbers
for column in data:
    data[column]=(data[column]-np.mean(data[column]))/np.std(data[column])
    
import sklearn.decomposition
pca = sklearn.decomposition.PCA(n_components=3)
principal_components = pca.fit(data).transform(data)

pl.figure('Reference Plot')
pl.scatter(principal_components[:, 0], principal_components[:, 1],principal_components[:, 2])

#decide how many clusters to choose, pick point at "elbow"
initial = [cluster.vq.kmeans(principal_components,i) for i in range(1,20)]
plt.plot([var for (cent,var) in initial])
plt.show()

#chose 10 clusters
kmeans = KMeans(n_clusters=10)
kmeans.fit(data)
pl.figure('K-means with 3 clusters')
pl.scatter(principal_components[:, 0], principal_components[:, 1],principal_components[:, 2], c=kmeans.labels_)
pl.show()

result=kmeans.labels_
resultdf2=pd.DataFrame(data=result)
resultdf2.to_csv("resultdf2.csv")

cent, var = initial[3]
#use vq() to get as assignment for each obs.
assignment,cdist = cluster.vq.vq(principal_components,cent)
plt.scatter(principal_components[:,0], principal_components[:,1],principal_components[:,2], c=assignment)
plt.show()
