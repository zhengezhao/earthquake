import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft

pi = np.pi

Fs = 80
Ts = 1.0/Fs

t = np.arange(0,6,Ts)
xt = np.sin(2*pi * t) + 0.3 * np.sin(2*pi * 7*t)

xf = fft(xt) / len(xt)
f = np.arange(len(xt), dtype=np.float64) / len(xt) * Fs

f,xf = f[:len(f)/2], xf[:len(xf)/2]

plt.subplot(211)
plt.plot(t, xt)

plt.subplot(212)
plt.stem(f, np.absolute(xf))

plt.show()

