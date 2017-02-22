
import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft
from time import time

for filenum in range(1,2):
	filename ='/Users/zhengezhao/Desktop/ArizonaPhDstudy/server/static/data/earthquakes/dataforEQ{0}/dataForSIM{0}'.format(str(filenum))+ '/1_ShearNormalized.txt'
	#df = pd.read_csv(filename, header=None,
     #                engine='python',sep=r"\s*")
	data = np.genfromtxt(filename)
	#data=data.reshape(-1)
	#data=data[::1000]
	# plt.subplot(211)
	# plt.plot(data[:,0])
	# plt.subplot(212)
	# plt.plot(data[:,1])

	# plt.show()

inner = []
#data=data.reshape(-1)
#data=data[:,0]
for i in range(int(len(data)/2)):
	x0 = np.roll(data,i,axis=0)
	#print (data.shape,x0.shape)
	#print(np.sum( x0*data ))
	#print (np.sum( x0*data ))

	#print(np.sum(x0*data,axis=0).shape)
	sumdot= np.sum(x0*data)
	#print (sumdot)
	inner.append(sumdot)
   
#make n>smallest period to filter it out

inner = np.array(inner)
#print (inner)
plt.plot(inner)
plt.show()	
n = 1000

ma = np.convolve(inner, np.ones([n,])/n,'same')
c1 = ma - np.roll(ma,1)>0
c2 = ma - np.roll(ma, -1)>0
c = np.logical_and(c1,c2)

plt.plot(ma)
plt.show()

'''
	Ts = 1
	Fs = 1.0/Ts

	n = len(data)
	xf = fft(data)
	# f - freq
	f = np.arange(n, dtype=np.float64) / n * Fs

	f,xf = f[:int(n/2)], xf[:int(n/2)]
	xf = np.absolute(xf)

'''