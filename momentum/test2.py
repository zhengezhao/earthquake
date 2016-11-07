import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import mpld3
from mpld3 import plugins, utils

i=0
ax=None
df = pd.read_csv('new.txt', header=None,
                     engine='python',sep=r"\s*")
            
pca = PCA(n_components=2)
pca.fit(df)
projected = [0,0]
for filename in ['1_MomentNormalized.txt', '2_MomentNormalized.txt']:
    df = pd.read_csv(filename, header=None,
                     engine='python',sep=r"\s*")
    projected[i] = pca.transform(df)
    t = np.arange(len(df))
    if ax:
        plt.subplot(1,2,i+1,sharex=ax,sharey=ax)
    else:
        ax=plt.subplot(1,2,i+1)
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.scatter(projected[i][:,0], projected[i][:,1],
                lw=0,
                c=t,
                cmap='cool')
    plt.colorbar()
    i+=1

plt.figure()
ax = plt.subplot(121)
plt.scatter(projected[0][1:,0]-projected[0][:-1,0], projected[0][1:,1]-projected[0][:-1,1])
plt.axis('equal')
plt.subplot(122, sharex=ax, sharey=ax)
plt.scatter(projected[1][1:,0]-projected[1][:-1,0], projected[1][1:,1]-projected[1][:-1,1])
plt.show()

