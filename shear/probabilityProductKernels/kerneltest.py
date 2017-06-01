import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft
from time import time

def GussianKernel(u1,reg1,u2,reg2):
	det1=np.linalg.det(reg1)
	det2=np.linalg.det(reg2)

	sigplus=np.linalg.inv( 1/2*np.linalg.inv(reg1)+1/2*np.linalg.inv(reg2) )

	uplus=0.5*np.linalg.inv(reg1).dot(u1)+0.5*np.linalg.inv(reg2).dot(u2)

	detplus=np.linalg.det(sigplus)

	K= det1**(-0.25)*det2**(-0.25)*detplus**(0.5)*\
	(np.exp(-0.25*u1.transpose().dot(np.linalg.inv(reg1)).dot(u1)\
		- 0.25*u2.transpose().dot(np.linalg.inv(reg2)).dot(u2)+ 0.5*uplus.transpose().dot(sigplus).dot(uplus)   ))
	return K[0,0]

'''
a = np.random.normal([0,0],1,[10000,2])
b = np.random.normal([0,0],10,[10000,2])

u1 = np.mean(a, axis=0).reshape([-1,1])
u2 = np.mean(b, axis=0).reshape([-1,1])

reg1 = np.cov(a.T)
reg2 = np.cov(b.T)

print(u1)
print(u2)
print(reg1)
print(reg2)
print( GussianKernel(u1,reg1, u2, reg2))
'''