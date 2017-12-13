import query as q 
import numpy as np

v1 = np.array(range(10)).reshape(5,-1)
v2 = np.array(range(10)).reshape(5,-1)
v3= np.ones(shape=(5,2))
v4 = np.concatenate((v2,v3),axis=1)
print(v1,v4)

print(q.maxshiftinnerproductv2(v4,v1))
