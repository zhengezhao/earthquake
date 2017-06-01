import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft
from time import time

data1= np.genfromtxt('distanceforequakesusedkernel.txt', delimiter=',')

data2=np.genfromtxt('d1.txt',delimiter=',')

plt.plot(1-data1,data2,'.',c='black')

plt.show()