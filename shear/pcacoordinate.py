
import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft
from time import time	

earthquakeperiod=[]
for filenum in range(1,51):
	filename ='/Users/zhengezhao/Desktop/ArizonaPhDstudy/server/static/data/earthquakes/dataforEQ{0}/dataForSIM{0}'.format(str(filenum))+ '/1_ShearNormalized.txt'
	#df = pd.read_csv(filename, header=None,
     #                engine='python',sep=r"\s*")
	data = np.genfromtxt(filename)
	#print data.shape

	#data=data.reshape(-1)

	data=np.sum(data,axis=1)
	#print (data.shape)
	Ts = 1
	Fs = 1.0/Ts

	n = len(data)
	xf = fft(data)
	# f - freq
	f = np.arange(n, dtype=np.float64) / n * Fs

	f,xf = f[:int(n/2)], xf[:int(n/2)]
	xf = np.absolute(xf)

	#print(xf.shape)
	#print(xf[:20])
	period=int(1/f[np.argmax(xf)])
	#print (period)
	earthquakeperiod.append(period)


for filenum in range(1,51):
	filename='/Users/zhengezhao/Desktop/ArizonaPhDstudy/git_zhenge/earthquake/shear/newkernel/segmentssimilarity,r=0.0001/segmentssimilarityforEQ{0}EQ{0}'.format(str(filenum))+'.txt'
	sim=np.genfromtxt(filename,delimiter=',')
	print (sim.shape)

	rowmean=sim.mean(axis=0)



	sim=sim-rowmean
	colmean=sim.mean(axis=1)
	sim=sim-colmean.reshape([sim.shape[0],1])

	#print(sim)

	U, s, V = np.linalg.svd(sim, full_matrices=True)

	#print (U.shape,s.shape,V.shape)


	s=s**0.5
	s=s.reshape([-1,1])
	s=np.diagflat(s)
	pca= U.dot(s)

	#print(time()-t0)
	print (pca.shape)

	np.savetxt('/Users/zhengezhao/Desktop/ArizonaPhDstudy/git_zhenge/earthquake/shear/newkernel/segmentscordinate/segmentscoordinateforEQ{0}'.format(str(filenum))+'.txt',pca,delimiter=',',fmt='%1.4f')
# 	plt.plot( pca[:,0],pca[:,1],".",color=(0.5/float(filenum),0.8/float(filenum),0.1))

# plt.show()


# '''
# inner = []
# data=np.reshape(data,(1,-1))
# for i in range(int(len(data)/2)):
#     x0 = np.roll(data, i)
#     inner.append( np.sum(x0*data, axis=0)  )

# inner = np.array(inner)
# print(inner.shape)

	
# '''
# inner = []
# for i in range(int(len(data)/2)):
#     x0 = np.roll(data, i)
#     inner.append(x0.dot(data))
   
# #make n>smallest period to filter it out

# inner = np.array(inner)
# plt.plot(inner)
# plt.show()	
# n = 1000

# ma = np.convolve(inner, np.ones([n,])/n,'same')
# c1 = ma - np.roll(ma,1)>0
# c2 = ma - np.roll(ma, -1)>0
# c = np.logical_and(c1,c2)

# plt.plot(ma)
# plt.show()
# #period = np.argmax(inner) 
# #period = int(period)

# #print (period)
# '''



'''
plt.subplot(211)
plt.plot(data[::100])
print(period)

plt.subplot(212)
plt.plot(f, np.absolute(xf))
print(period)

plt.show()
print('showed')
'''

#similarity matrix
'''
fig = plt.figure(figsize=(8,8))
plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0,hspace=0,wspace=0)


#np.savetxt('simforcarlos.txt',sim,delimiter=',',fmt='%1.4e')
#np.savetxt('segmentsforcarlos.txt',segments,delimiter=',',fmt='%1.7e')


for i,v1 in enumerate(segments):

	v1 = v1[:len(v1)-len(v1)%13]
	v1 = v1.reshape([-1,13])

	v1 = np.sum(v1, axis=1)/13

	ax = fig.add_subplot(segments.shape[0]+1,segments.shape[0]+1,i+2)
	ax.plot(v1[::40])
	ax.set_xticks([]) 
	ax.set_yticks([]) 
	ax = fig.add_subplot(segments.shape[0]+1,segments.shape[0]+1,(i+1)*(segments.shape[0]+1)+1)
	ax.plot(v1[::40])
	ax.set_xticks([]) 
	ax.set_yticks([])




ax = plt.subplot2grid([segments.shape[0]+1,segments.shape[0]+1],[1,1],rowspan=segments.shape[0],colspan=segments.shape[0])
ax.matshow(1-sim, cmap=plt.cm.gray)
ax.set_xticks([]) 
ax.set_yticks([])
'''
#plt.savefig('foo.png',dpi=330)