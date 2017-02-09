import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft


xt = np.loadtxt('1_ShearNormalized.txt')
xt = xt[:,5]
Ts = 1
Fs = 1.0/Ts

n = len(xt)
xf = fft(xt)
# f - freq
f = np.arange(n, dtype=np.float64) / n * Fs

f,xf = f[:int(n/2)], xf[:int(n/2)]
xf = np.absolute(xf)

print (1/f[np.argmax(xf)])

plt.subplot(311)
plt.plot(np.arange(n)[::100], xt[::100])
plt.xlabel('time /s')

plt.subplot(312)
plt.stem(f[10:int(n/10):3], xf[10:int(n/10):3])
plt.xlabel('freq /Hz')

plt.subplot(313)
plt.stem(1/f[10:int(n/10):3], xf[10:int(n/10):3])
plt.xlabel('period /s')

plt.show()
