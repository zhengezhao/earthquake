'''
USAGE
pyhton period2 filename.txt floor_index
 
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft



filename = sys.argv[1]
#floor = int(sys.argv[2])

data = np.genfromtxt(filename)
#data = data[:,floor])

def maxshiftinnerproduct(v1,v2):
	x1 = v1[::int(len(v1)/49)]
	x2 = v2[::int(len(v2)/49)]
	
	inner=[]
	for i in range(len(x1)):
		x1=np.roll(x1,i)
		inner.append(x1.dot(x2)/(np.linalg.norm(x1)*(np.linalg.norm(x2))))
	return np.amax(inner)

'''
inner = []
data=np.reshape(data,(1,-1))
for i in range(int(len(data)/2)):
    x0 = np.roll(data, i)
    inner.append( np.sum(x0*data, axis=0)  )

inner = np.array(inner)
print(inner.shape)'''

data=data.reshape(-1)
'''
inner = []
for i in range(int(len(data)/2)):
    x0 = np.roll(data, i)
    inner.append(x0.dot(data))
   
#make n>smallest period to filter it out

inner = np.array(inner)
plt.plot(inner)
plt.show()	
n = 1000

ma = np.convolve(inner, np.ones([n,])/n,'same')
c1 = ma - np.roll(ma,1)>0
c2 = ma - np.roll(ma, -1)>0
c = np.logical_and(c1,c2)

plt.plot(ma)
plt.show()
#period = np.argmax(inner) 
#period = int(period)

#print (period)
'''


Ts = 1
Fs = 1.0/Ts

n = len(data)
xf = fft(data)
# f - freq
f = np.arange(n, dtype=np.float64) / n * Fs

f,xf = f[:int(n/2)], xf[:int(n/2)]
xf = np.absolute(xf)

print(xf.shape)
period=int(1/f[np.argmax(xf)])



segments=np.reshape(data[:-(len(data)%period)],[-1,period])
sim = np.ones(shape=(segments.shape[0],segments.shape[0]))

for i,v1 in enumerate(segments):
	for j,v2 in enumerate(segments):
		sim[i,j]=maxshiftinnerproduct(v1,v2)

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


'''
for i in range(sim.shape[0]):
	for j in range(sim.shape[1]):
		ax = plt.subplot(segments.shape[0]+1,segments.shape[0]+1, 31 + i*30 + j+1)
		ax.text(str(sim[i,j]), 0,0)
		ax.set_xticks([]) 
		ax.set_yticks([])
'''

ax = plt.subplot2grid([segments.shape[0]+1,segments.shape[0]+1],[1,1],rowspan=segments.shape[0],colspan=segments.shape[0])
ax.matshow(1-sim, cmap=plt.cm.gray)
ax.set_xticks([]) 
ax.set_yticks([])

#plt.savefig('foo.png',dpi=330)

plt.show()






