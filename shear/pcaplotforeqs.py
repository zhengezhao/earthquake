import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import manifold
from sklearn.metrics import euclidean_distances



distance=np.genfromtxt('distanceforequakes.txt',delimiter=',')
distance=np.array(distance)
data=distance


matrix=np.ones(shape=(50,50))

a= np.linalg.norm(data)**2
for i in range(matrix.shape[0]):
	b= np.linalg.norm(data[i]) **2

	for j in range(matrix.shape[0]):
		c=np.linalg.norm(data[:,j]) **2

		matrix[i][j]= -0.5*(data[i][j]**2-c/matrix.shape[0]-b/matrix.shape[0]+a/(matrix.shape[0]*matrix.shape[0]))


U, s, V = np.linalg.svd(matrix, full_matrices=True)

s=s**0.5
s=s.reshape([-1,1])
s=np.diagflat(s)
pca= U.dot(s)


a=np.linalg.norm(pca[:,0])**2
b=np.linalg.norm(pca[:,1]) **2

c=np.linalg.norm(pca)**2

print (a,b,c,(a+b)/c)


N = pca.shape[0]

labels = ['{0}'.format(i+1) for i in range(N)]


plt.subplots_adjust(bottom = 0.1)
plt.scatter(
     pca[:, 0], pca[:, 1], marker='o')

for label, x, y in zip(labels, pca[:, 0], pca[:, 1]):
	plt.text(x,y,label,color='k',fontsize=6)

# plt.subplots_adjust(bottom = 0.1)
# plt.scatter(
#     pca[:, 0], pca[:, 1], marker='o')

# for label, x, y in zip(labels, pca[:, 0], pca[:, 1]):
#     plt.annotate(
#         label,
#         xy=(x-30, y-16), xytext=(-10, 10),
#         textcoords='offset points',
#         arrowprops=dict(arrowstyle = '-', connectionstyle='arc3,rad=0'))

plt.show()

