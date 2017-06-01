import numpy as np
import matplotlib.pyplot as plt
import sys
fig = plt.figure(figsize=(8,8))
plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0,hspace=0,wspace=0)


#np.savetxt('simforcarlos.txt',sim,delimiter=',',fmt='%1.4e')
#np.savetxt('segmentsforcarlos.txt',segments,delimiter=',',fmt='%1.7e')
# for filenum in range(1,51):
# 	filename ='/Users/zhengezhao/Desktop/ArizonaPhDstudy/server/static/data/earthquakes/dataforEQ{0}/dataForSIM{0}'.format(str(filenum))+ '/1_ShearNormalized.txt'

# for i,v1 in enumerate(segments):
# 	ax = fig.add_subplot(segments.shape[0]+1,segments.shape[0]+1,i+2)
# 	ax.plot(v1[::40])
# 	ax.set_xticks([]) 
# 	ax.set_yticks([]) 
# 	ax = fig.add_subplot(segments.shape[0]+1,segments.shape[0]+1,(i+1)*(segments.shape[0]+1)+1)
# 	ax.plot(v1[::40])
# 	ax.set_xticks([]) 
# 	ax.set_yticks([])


'''
for i in range(sim.shape[0]):
	for j in range(sim.shape[1]):
		ax = plt.subplot(segments.shape[0]+1,segments.shape[0]+1, 31 + i*30 + j+1)
		ax.text(str(sim[i,j]), 0,0)
		ax.set_xticks([]) 
		ax.set_yticks([])
'''

fig = plt.figure(figsize=(8,8))
plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0,hspace=0,wspace=0)


data=np.genfromtxt("newkernel/segmentssimilarity,r=0.00025/segmentssimilarityforEQ1EQ2.txt", delimiter=',');

segments=np.array(data)

ax = plt.subplot2grid([segments.shape[0],segments.shape[1]],[0,0], rowspan=segments.shape[0], colspan=segments.shape[1] )
ax.matshow(1-data, cmap=plt.cm.gray)
ax.set_xticks([]) 
ax.set_yticks([])

#plt.savefig('foo.png',dpi=330)

plt.show()
