import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft
from time import time
import matplotlib.cm as cm

def add_arrow(line, position=None, direction='right', size=30, color=None):
	if color is None:
		color = line.get_color()

	xdata = line.get_xdata()
	ydata = line.get_ydata()
	#print (xdata)
	#print (ydata)

	if position is None:
		position = xdata.mean()
	# find closest index
	# find closest index
	start_ind = np.argmin(np.absolute(xdata - position))
	if direction == 'right':
		end_ind = 1
	else:
		end_ind = 0
	#print (start_ind,end_ind)

	line.axes.annotate('',
		xytext=(xdata[start_ind], ydata[start_ind]),
		xy=(xdata[end_ind], ydata[end_ind]),
		arrowprops=dict(arrowstyle="->", color=color),
	size=size
	)



segments=[[] for i in range(2)]

#print (segments)

for j,filenum in enumerate([15,46]):
	filename ='/Users/zhengezhao/Desktop/ArizonaPhDstudy/git_zhenge/earthquake/shear/newkernel/segmentscordinate/segmentscoordinateforEQ{0}'.format(str(filenum))+ '.txt'
	data = np.genfromtxt(filename,delimiter=',')
	for i in range(data.shape[0]):
		#print ([data[i,0],data[i,1]])
		segments[j].append((data[i,0],data[i,1]) )

# x=np.arrage(50)
# ys=[i+x+(i*x)**2 for i in range(50)]
#segments=segments[:1]
#print (len(segments))
colors = cm.rainbow(np.linspace(0, 1, len(segments)))

#print (colors.shape)
for s, c in zip(segments, colors):
	s= np.array(s)
	for i in range(s.shape[0]-2):
		x= s[i:i+2,0]
		y= s[i:i+2,1]
		#print (x,y)
		plt.scatter(x, y, s=250,color=c)
		#line=plt.plot(x,y,".",color=c)[0]
		#add_arrow(line)
plt.show()

