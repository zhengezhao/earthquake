import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

xt = np.linspace(0,10000)/1000
xt = np.sin(xt) + 0.2*np.sin(0.01*xt)

ft=fft(xt)

plt.subplot(211)
plt.plot(xt)
plt.subplot(212)
plt.stem(np.absolute(ft))
plt.show()

