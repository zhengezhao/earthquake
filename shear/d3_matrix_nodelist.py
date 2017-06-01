import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft
from time import time	
import numpy as np
from sklearn.decomposition import PCA

distance=np.genfromtxt('distanceforequakes.txt',delimiter=',')
distance=np.array(distance)
mind=np.amin(distance)
maxd=np.amax(distance)

normalizedsim=1.0-(distance-mind)/(maxd-mind)

array=[]


# for i in range(50):
# 	row="eq"+str(i+1)
# 	array.append(row)

# with open('id_eq.csv','w') as fo:
# 	fo.write("id"+'\n')
# 	for i in array:
# 		fo.write(i+'\n')

for i in range(50):
	newcsv=[]
	for j in range(50):
		column="eq"+str(j+1)
		sim=str(normalizedsim[i,j])
		newcsv.append((column,sim))
		newcsv = sorted(newcsv, key=itemgetter(1,0))
		newcsv=newcsv[1:]
	with open('eq{0}'.format(str(i+1))+'1.csv','w') as fo:
		fo.write('date'+','+'value'+','+'\n')
		for a in newcsv:
			fo.write(a[0]+','+a[1]+'\n')




# np.savetxt('segmentssimilarityforEQ{0}EQ{1}'.format(str(filenum1),str(filenum2))+'.txt',sim,delimiter=',',fmt='%1.4f')

# for i in range(50):
# 	row= "eq"+str(i+1)
# 	for j in range(50):
# 		column="eq"+str(j+1)
# 		sim=str(normalizedsim[i,j])
# 		array.append((row,column,sim))




# with open('node_list_eq.txt','w') as fo:
# 	for i in array:
# 		fo.write(i[0]+','+i[1]+','+i[2]+'\n')