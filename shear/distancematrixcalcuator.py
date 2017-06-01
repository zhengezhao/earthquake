import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft
from time import time	

def innerproductofDataset(d1,d2,s):
	innerproductsum=0
	n1=d1.shape[0]
	n2=d2.shape[0]
	a=1
	b=1
	for i in range(n1):
		for j in range(n2):
			innerproductsum=innerproductsum+a*b*s[i,j]
	return (innerproductsum / (n1*n2))


def DatasetDistance(d1,d2,s1,s2,s12):
	d=innerproductofDataset(d1,d1,s1)+innerproductofDataset(d2,d2,s2)-2*innerproductofDataset(d1,d2,s12)
	# print (d**(1/2))
	return (d**(1/2))
	#return np.argmax(xt),np.amax(xt)
	#return np.amax(xt)/(np.linalg.norm(x1)*(np.linalg.norm(x2)))*(np.linalg.norm(v1)*(np.linalg.norm(v2)))

# def DatasetDistance(d1,d2):
# 	sum=0
# 	n1=d1.shape[0]
# 	n2=d2.shape[0]
# 	for i in range(n1):
# 		for j in range(n2):
# 			sum=sum+(np.linalg.norm(d1[i]-d2[j]))**2

# 	return sum/(n1*n2)



earthquakeperiod=[]
distance=np.ones(shape=(50,50))



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


for filenum1 in range(1,51):
	filename1 ='/Users/zhengezhao/Desktop/ArizonaPhDstudy/server/static/data/earthquakes/dataforEQ{0}/dataForSIM{0}'.format(str(filenum1))+ '/1_ShearNormalized.txt'
	sim1file='/Users/zhengezhao/Desktop/ArizonaPhDstudy/git_zhenge/earthquake/shear/segmentssimilarity/segmentssimilarityforEQ{0}EQ{1}'.format(str(filenum1),str(filenum1))+'.txt'
	sim1=np.genfromtxt(sim1file,delimiter=',')
	period1=13*earthquakeperiod[filenum1-1]
	data1 = np.genfromtxt(filename1)
	data1=data1.reshape(-1)
	if (len(data1)%period1) !=0:
		segments1=np.reshape(data1[:-(len(data1)%period1)],[-1,period1])
	else:
		segments1=np.reshape(data1,[-1,period1])
	for filenum2 in range(filenum1,51):
		sim2file='/Users/zhengezhao/Desktop/ArizonaPhDstudy/git_zhenge/earthquake/shear/segmentssimilarity/segmentssimilarityforEQ{0}EQ{1}'.format(str(filenum2),str(filenum2))+'.txt'
		sim2=np.genfromtxt(sim2file,delimiter=',')
		sim12file='/Users/zhengezhao/Desktop/ArizonaPhDstudy/git_zhenge/earthquake/shear/segmentssimilarity/segmentssimilarityforEQ{0}EQ{1}'.format(str(filenum1),str(filenum2))+'.txt'
		sim12=np.genfromtxt(sim12file,delimiter=',')
		filename2 ='/Users/zhengezhao/Desktop/ArizonaPhDstudy/server/static/data/earthquakes/dataforEQ{0}/dataForSIM{0}'.format(str(filenum2))+ '/1_ShearNormalized.txt'
		period2=13*earthquakeperiod[filenum2-1]
		mperiod=max(period1,period2)
		data2 = np.genfromtxt(filename2)
		data2=data2.reshape(-1)
		if (len(data2)%period2) !=0:
			segments2=np.reshape(data2[:-(len(data2)%period2)],[-1,period2])
		else:
			segments2=np.reshape(data2,[-1,period2])
		ussegments1=np.ones(shape=(segments1.shape[0],mperiod))
		for i in range(segments1.shape[0]):
			ussegments1[i]=np.interp(np.linspace(0,1,mperiod),np.linspace(0,1,period1),segments1[i])
		ussegments2=np.ones(shape=(segments2.shape[0],mperiod))
		for i in range(segments2.shape[0]):
			ussegments2[i]=np.interp(np.linspace(0,1,mperiod),np.linspace(0,1,period2),segments2[i])

		d=DatasetDistance(ussegments1,ussegments2,sim1,sim2,sim12)
		#print (d,period1,period2,segments1.shape,ussegments1.shape,segments2.shape,ussegments2.shape)
		distance[filenum1-1,filenum2-1]=d
		distance[filenum2-1,filenum1-1]=d

		#print (sim.shape,period1,period2,segments1.shape,ussegments1.shape,segments2.shape,ussegments2.shape)

np.savetxt('distanceforequakesusedkernel.txt',distance,delimiter=',',fmt='%1.4f')
print (d.shape)
#similarity matrix



fig = plt.figure(figsize=(8,8))
plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0,hspace=0,wspace=0)





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

#plt.savefig('foo.png',dpi=330)

plt.show()