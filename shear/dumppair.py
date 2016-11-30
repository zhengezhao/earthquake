import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import json





df = pd.read_csv('new.txt', header=None,
                     engine='python',sep=r"\s*")
            
pca = PCA(n_components=2)
pca.fit(df)

projected = [0,0]
for filename in ['1_ShearNormalized.txt', '2_ShearNormalized.txt']:
    df = pd.read_csv(filename, header=None,
                     engine='python',sep=r"\s*")
    x = pca.transform(df)
    v1,v2 = pca.components_
    print (v1,v2)
    v = x[1:,:]-x[:-1,:]
    pca2 = PCA(n_components=2)
    pca2.fit(v)

    v=pca2.transform(v)
    transform = pca.inverse_transform(v)
    dumped = [{'v':list(v[i,:]), 'vCap':list(transform[i,:])} for i in range(v.shape[0])]
    f = open(filename.split('.')[0]+'pair.js', 'w')
    f.write('var data'+filename.split('_')[0]+' = ')
    json.dump(dumped, f,indent=1)



