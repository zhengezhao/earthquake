import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft

def maxshiftrinnerproduct(v1,v2):
	if v1.ndim>1:
		v1=np.sum(v1,axis=1)
		v2=np.sum(v2,axis=1)
	Ts=1
	Fs=1.0/Ts
	n=len(v1)
	f = np.arange(n, dtype=np.float64) / n * Fs
	xf1=fft(v1)
	xf2=fft(v2[::-1])
	product=xf1*xf2
	xt=np.fft.ifft(product)
	xt=xt.real
	#return np.argmax(xt),np.amax(xt)
	return np.amax(xt)/(np.linalg.norm(v1)*(np.linalg.norm(v2)))

print (maxshiftrinnerproduct(np.arange(6).reshape(2,3),np.array([3,4,5,0,1,2]).reshape(2,3) ))

pi = np.pi

Fs = 80
Ts = 1.0/Fs

t = np.arange(0,6,Ts)
xt = np.sin(2*pi * t) + 0.3 * np.sin(2*pi * 7*t)

#print (len(xt),len(t))

xf = fft(xt)

inversexf=np.fft.ifft(xf)
plt.subplot(311)
plt.plot(t,np.absolute(inversexf))
plt.subplot(312)
plt.plot(t,inversexf.real)
plt.subplot(313)
plt.plot(t,xt)
#plt.show()



f = np.arange(len(xt), dtype=np.float64) / len(xt) * Fs


f,xf = f[:int(len(f)/2)], xf[:int(len(xf)/2)]
xf=np.absolute(xf)
#print (np.argmax(xf),f)
period = int(1/f[np.argmax(xf)])
#print (period,f[np.argmax(xf)])

plt.subplot(211)
plt.plot(t, xt)

plt.subplot(212)
plt.stem(f, np.absolute(xf))

#plt.show()

