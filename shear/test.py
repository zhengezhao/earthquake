"""
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
'''with open('1_ShearNormalized.txt', 'r') as f:  
    data = f.readlines()
    x = np.array(None)  x.i
    for line in data:
        #print (line)  
        dom = line.split()         
        numbers_float = map(float, dom) 	
        l = list(numbers_float)
        x.insert(l)'''
df = pd.read_csv('1_ShearNormalized.txt', header = None ,sep=r"\s*")

#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#print (x)	
pca = PCA(n_components=2)
pca.fit(df)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
print(pca.explained_variance_ratio_)
projected = pca.transform(df)
#plt.subplot(121)
plt.xlim([-10,10])
plt.ylim([-10,10])

plt.plot(projected[:,0], projected[:,1],'-',alpha = 0.3)




df = pd.read_csv('2_ShearNormalized.txt', header = None ,sep=r"\s*")
pca = PCA(n_components=2)
pca.fit(df)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
print(pca.explained_variance_ratio_)
projected = pca.transform(df)
plt.subplot(122)
plt.xlim([-10,10])
plt.ylim([-10,10])

#plt.scatter(projected[:,0], projected[:,1],alpha = 0.3)

#plt.show()

x = np.linspace(0,4*np.pi, 100)
y = np.sin(x)

#c    is your color value, in your case time,
#cmap is colormap, serach for different colormaps
#s    is size of markers
plt.scatter(x,y,c=x, linewidth=0, s=30, cmap='gnuplot')
plt.colorbar()

plt.show()
"""

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

ax = None



for i,filename in enumerate(['1_ShearNormalized.txt', '2_ShearNormalized.txt']):
    df = pd.read_csv(filename, header=None,
                     engine='python',sep=r"\s*")
            
    pca = PCA(n_components=3)
    pca.fit(df)
    projected = pca.transform(df)
    t = np.arange(len(df))
    print (t)
    if ax:
        plt.subplot(2,2,i+1,sharex=ax)
    else:
	    ax=plt.subplot(2,2,i+1)
    #plt.xlim([-10,10])
    #plt.ylim([-10,10])
    #plt.scatter(projected[:,0], t 
    #            )
    #curve
    
    print(np.dot(projected[:,0],projected[:,1]))
    plt.plot(t,projected[:,0],"r-",alpha=0.3)
    plt.plot(t,(1 if i == 1 else -1)*projected[:,1],"b-",alpha=0.3)
    
    plt.plot(t,projected[:,0]*projected[:,1],"k-",alpha=0.8)
    plt.plot([0,len(df)],[0,0],"k-",alpha=0.3)
    #plt.colorbar()

    v1,v2,v3 = pca.components_
    plt.subplot(2,2,i+3)
    plt.plot(v1 if i==0 else -v1,np.arange(13))

    plt.plot(v2,np.arange(13))
    plt.plot(v3,np.arange(13))

plt.show()
