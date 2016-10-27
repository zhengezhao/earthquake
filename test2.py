import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

i=0
ax=None
df = pd.read_csv('new.txt', header=None,
                     engine='python',sep=r"\s*")
            
pca = PCA(n_components=2)
pca.fit(df)

for filename in ['1_ShearNormalized.txt', '2_ShearNormalized.txt']:
    df = pd.read_csv(filename, header=None,
                     engine='python',sep=r"\s*")
    projected = pca.transform(df)
    t = np.arange(len(df))
    if ax:
        plt.subplot(1,2,i+1,sharex=ax,sharey=ax)
    else:
        ax=plt.subplot(1,2,i+1)
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.scatter(projected[:,0], projected[:,1],
                lw=0,
                c=t,
                cmap='cool')
    plt.colorbar()
    i+=1
plt.show()
