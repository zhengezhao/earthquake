import pandas as pd

import json
from glob import glob
import scipy,sys
import numpy as np
from scipy.stats import pearsonr
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt

'''
for filenum in range(1,51):
	filename ='C:\\Users\\ZhengeZhao\\Desktop\\ArizonaPhDstudy\\server\\static\\data\\earthquakes\\dataforEQ{0}\\dataForSIM{0}'.format(str(filenum))+ '/1_ShearNormalized.txt'
	df = pd.read_csv(filename, header=None,
                     engine='python',sep=r"\s*")
	#print (df[5])

	array = np.array(df[5])
	lst = []

	for i in range(len(array)):
		temp=np.inner(array,np.roll(array,i+1)) 
		lst.append(str(temp))

	with open('period.txt','a') as fo:
		for i in lst:
			fo.write(i+' ')
		fo.write('\n')
'''
'''
df = pd.read_csv('1_ShearNormalized.txt', header=None,
                     engine='python',sep=r"\s*")
	#print (df[5])

array = np.array(df[5])
lst = []

for i in range(len(array)):
	temp=np.inner(array,np.roll(array,i+1)) 
	lst.append(str(temp))

#print (lst)
#plt.plot(np.fft.fft(lst),".-")

plt.plot(lst,".-")
plt.show()
'''
with open('period.txt') as f:
	for i in f.readlines():
		i= i.replace('\n','').split(' ')[:-1]
		#print (i)
		i= [float(j) for j in i][:int(len(i)/2)]
		plt.plot(i,alpha=0.7)
plt.show()
#print (df)

#plt.plot(df)
#plt.show()
'''
#find the period 
for i in range(10,len(lst)):
	for j in range(10):
		if lst[i+1] >= lst[i-j] and lst[i+1] >= lst[i+2+j]:
			pass
		else:
			break
	if j ==9:		
		print (i,lst[i])
		break
'''