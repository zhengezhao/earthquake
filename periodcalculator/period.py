import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft
from time import time

earthquakeperiod = []

for filenum in range(1, 51):

    # the path of the filename should be changed to your data
    filename = '/Users/zhengezhao/Desktop/ArizonaPhDstudy/server/static/data/earthquakes/dataforEQ{0}/dataForSIM{0}'.format(str(filenum)) + '/1_ShearNormalized.txt'

    data = np.genfromtxt(filename)
    # print data.shape

    # data=data.reshape(-1)

    data = np.sum(data, axis=1)
    #print (data.shape)
    Ts = 1
    Fs = 1.0 / Ts

    n = len(data)
    xf = fft(data)
    # f - freq
    f = np.arange(n, dtype=np.float64) / n * Fs

    f, xf = f[:int(n / 2)], xf[:int(n / 2)]
    xf = np.absolute(xf)

    # print(xf.shape)
    # print(xf[:20])
    period = int(1 / f[np.argmax(xf)])
    #print (period)
    earthquakeperiod.append(period)

print (earthquakeperiod)
plt.hist(earthquakeperiod)
plt.show()
