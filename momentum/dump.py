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
for filename in ['1_MomentNormalized.txt', '2_MomentNormalized.txt']:
    df = pd.read_csv(filename, header=None,
                     engine='python',sep=r"\s*")
    x = pca.transform(df)
    f = open(filename.split('.')[0]+'.js', 'w')
    f.write('var data'+filename.split('_')[0]+' = ')
    json.dump(list(zip(list(x[:,0]),list(x[:,1]))), f)



