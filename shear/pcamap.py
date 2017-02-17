import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft
from time import time
import matplotlib.cm as cm

segments=[[] for i in range(50)]

#print (segments)

for filenum in range(1,51):
	filename ='/Users/zhengezhao/Desktop/ArizonaPhDstudy/git_zhenge/earthquake/shear/segmentscoordinateforEQ{0}'.format(str(filenum))+ '.txt'
	data = np.genfromtxt(filename,delimiter=',')
	for i in range(data.shape[0]):
		#print ([data[i,0],data[i,1]])
		segments[filenum-1].append((data[i,0],data[i,1]) )

# x=np.arrage(50)
# ys=[i+x+(i*x)**2 for i in range(50)]
segments=segments[:10]
print (len(segments))
colors = cm.rainbow(np.linspace(0, 1, len(segments)))

#print (colors.shape)
for y, c in zip(segments, colors):
# plt.scatter(y[0][:,0], y[0][:,1], color=c)
	plt.plot(y,".",color=c)
plt.show()

