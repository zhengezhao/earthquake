import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft
from time import time	
import numpy as np
from sklearn.decomposition import PCA


earthquakeperiod=[]
earthquakes=np.ones(shape=(50,50))



distance=np.genfromtxt('distanceforequakes.txt',delimiter=',')
distance=np.array(distance)
mind=np.amin(distance)
maxd=np.amax(distance)
normalizeddistance=(distance-mind)/(maxd-mind)

fig = plt.figure(figsize=(8,8))
plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0,hspace=0,wspace=0)




for filenum in range(1,51):
	filename ='/Users/zhengezhao/Desktop/ArizonaPhDstudy/server/static/data/earthquakes/dataforEQ{0}/dataForSIM{0}'.format(str(filenum))+ '/1_ShearNormalized.txt'

	data = np.genfromtxt(filename)
	pca = PCA(n_components=1)
	pca.fit(data)
	projected=pca.transform(data)
	v1=projected[:,0]
	ax=fig.add_subplot(earthquakes.shape[0]+1,earthquakes.shape[0]+1,filenum+1)
	ax.plot(v1[::400])
	ax.set_xticks([]) 
	ax.set_yticks([])
	ax = fig.add_subplot(earthquakes.shape[0]+1,earthquakes.shape[0]+1,(filenum)*(earthquakes.shape[0]+1)+1)
	ax.plot(v1[::400])
	ax.set_xticks([]) 
	ax.set_yticks([])



ax = plt.subplot2grid([earthquakes.shape[0]+1,earthquakes.shape[0]+1],[1,1],rowspan=earthquakes.shape[0],colspan=earthquakes.shape[0])
ax.matshow(normalizeddistance, cmap=plt.cm.gray)
ax.set_xticks([]) 
ax.set_yticks([])

plt.show()

