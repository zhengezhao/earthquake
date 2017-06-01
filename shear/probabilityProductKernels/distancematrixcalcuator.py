import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft
from time import time
from sklearn.decomposition import PCA

# def innerproductofDataset(d1,d2,s):
# 	innerproductsum=0
# 	n1=d1.shape[0]
# 	n2=d2.shape[0]
# 	a=1
# 	b=1
# 	for i in range(n1):
# 		for j in range(n2):
# 			innerproductsum=innerproductsum+a*b*s[i,j]*(np.linalg.norm(d1[i])*np.linalg.norm(d2[j]))
# 	return (innerproductsum / (n1*n2))


def Gaussian(f, e_num, ita):
    # k=n d=5
    k = f.shape[0]
    d = f[0].shape[0]
    u = np.sum(f, axis=0) / k
    u = u.reshape(-1, 1)
    sig = np.zeros(shape=(d, d))
    for i in range(k):
        sig += (f[i].reshape(-1, 1) - u) * ((f[i].reshape(-1, 1) - u).transpose())
    sig = sig / k
    return [u, sig]

    # sig=np.zeros(shape=(d,d))
    # reg=np.zeros(shape=(d,d))
    # #print (sig.shape, u.shape, f[0].shape)
    # for i in range(k):
    # 	# sig+=(f[i]-u)*((f[i]-u).reshape((-1,1)))
    # 	sig+=(f[i].reshape(-1,1)-u)*((f[i].reshape(-1,1)-u).transpose())
    # sig=sig/k
    # U, s, V = np.linalg.svd(sig, full_matrices=True)
    # for i in range(e_num):
    # 	reg+=U[:,i].reshape(-1,1)*s[i]*U[:,i].reshape(-1,1).transpose()
    # reg+=ita*np.identity(d)
    # return [u,reg]
# def Gaussian(f,e_num,ita):
# 	k=f.shape[0]
# 	d=f[0].shape[0]
# 	u=np.sum(f,axis=0)/k
# 	u=u.reshape(-1,1)
# 	sig=np.zeros(shape=(d,d))
# 	# reg=np.zeros(shape=(d,d))
# 	#print (sig.shape, u.shape, f[0].shape)
# 	for i in range(k):
# 		# sig+=(f[i]-u)*((f[i]-u).reshape((-1,1)))
# 		sig+=(f[i].reshape(-1,1)-u)*((f[i].reshape(-1,1)-u).transpose())
# 	sig=sig/k
# 	# print (k,d)
# 	# print (u.reshape(1,-1))
# 	# print (sig)
# 	# U, s, V = np.linalg.svd(sig, full_matrices=True)
# 	# for i in range(e_num):
# 	# 	reg+=U[:,i].reshape(-1,1)*s[i]*U[:,i].reshape(-1,1).transpose()
# 	# reg+=ita*np.identity(d)
# 	return [u,sig]


def GussianKernel(u1, reg1, u2, reg2):
    det1 = np.linalg.det(reg1)
    det2 = np.linalg.det(reg2)
    sigplus = np.linalg.inv(1 / 2 * np.linalg.inv(reg1) + 1 / 2 * np.linalg.inv(reg2))
    uplus = 0.5 * np.linalg.inv(reg1).dot(u1) + 0.5 * np.linalg.inv(reg2).dot(u2)
    detplus = np.linalg.det(sigplus)
    # print(det1,det2,detplus)
    K = det1**(-0.25) * det2**(-0.25) * detplus**(0.5) *\
        (np.exp(-0.25 * u1.transpose().dot(np.linalg.inv(reg1)).dot(u1)
                - 0.25 * u2.transpose().dot(np.linalg.inv(reg2)).dot(u2) + 0.5 * uplus.transpose().dot(sigplus).dot(uplus)))
    return K[0, 0]


def DatasetDistance(d1, d2, s1, s2, s12):
    # d1.shape=(numberofsegmentsXthe dimension of each segnments )
    # d2.shape=(numberofsegmentsXthe dimension of each segnments )
    # s1: the simliarity matrix for d1Xd1
    # s2: the similiarity matrix for d2Xd2
    # s12: the similarity matrix for d1Xd2
    ita = 0.1
    e_num = 5
    numOfd1 = d1.shape[0]
    numOfd2 = d2.shape[0]
    numOfFeature = d1.shape[0] + d2.shape[0]

    kmatrix = np.ones(shape=(numOfFeature, numOfFeature))
    # pca_kmatrix= np.ones(shape=(numOfFeature,numOfFeature))
    for i in range(numOfFeature):
        for j in range(numOfFeature):
            if(i < numOfd1 and j < numOfd1):
                kmatrix[i, j] = s1[i, j]
            elif(i >= numOfd1 and j >= numOfd1):
                kmatrix[i, j] = s2[i - numOfd1, j - numOfd1]
            elif(i < numOfd1 and j >= numOfd1):
                kmatrix[i, j] = s12[i, j - numOfd1]
            else:
                kmatrix[i, j] = s12[j, i - numOfd1]

    # rowmean=kmatrix.mean(axis=0)
    # kmatrix=kmatrix-rowmean
    # colmean=kmatrix.mean(axis=1)
    # kmatrix=kmatrix-colmean.reshape([kmatrix.shape[0],1])

    for i in range(kmatrix.shape[0]):
        for j in range(kmatrix.shape[1]):
            kmatrix[i, j] = np.exp(kmatrix[i, j])

    U, s, V = np.linalg.svd(kmatrix, full_matrices=True)
    s = s**0.5
    s = s.reshape([-1, 1])
    s = np.diagflat(s)
    Kprime = U.dot(s)
    pca = PCA(n_components=e_num)
    pca.fit(Kprime)
    projected = pca.transform(Kprime)

    # for i in range(len(s)):
    # 	if i>=5:
    # 		s[i]=ita
    # 	# else:
    # 	# 	s[i]=s[i]+ita
    # print (s)
    # s=s.reshape([-1,1])
    # s=np.diagflat(s)
    # pca= U.dot(s)

    # print(pca)

    # f1=pca[:numOfd1]
    # f2=pca[numOfd1:]

    f1 = projected[:numOfd1]
    f2 = projected[numOfd1:]
    u1, reg1 = Gaussian(f1, e_num, ita)
    # print(reg1)
    u2, reg2 = Gaussian(f2, e_num, ita)
    #print (u1,reg1)

    return GussianKernel(u1, reg1, u2, reg2)

    # print(s)
    # b=s[0]
    # c=s[1]+s[2]+s[3]+s[4]
    # a=[np.sum(s[:i])/ np.sum(s) for i in range(len(s))]
    # print (a)
    # plt.plot(a)
    # plt.show()

    # d=innerproductofDataset(d1,d1,s1)+innerproductofDataset(d2,d2,s2)-2*innerproductofDataset(d1,d2,s12)
    # print (d**(1/2))
# return (d**(1/2))


earthquakeperiod = []
distance = np.ones(shape=(50, 50))


for filenum in range(1, 51):
    filename = '/Users/zhengezhao/Desktop/ArizonaPhDstudy/server/static/data/earthquakes/dataforEQ{0}/dataForSIM{0}'.format(str(filenum)) + '/1_ShearNormalized.txt'
    # df = pd.read_csv(filename, header=None,
 #                engine='python',sep=r"\s*")
    data = np.genfromtxt(filename)
    # print data.shape

    # data=data.reshape(-1)

    data = np.sum(data, axis=1)
    #print (data.shape)
    Ts = 1
    Fs = 1.0 / Ts

    n = len(data)
    xf = fft(data)
    # f - freq
    f = np.arange(n, dtype=np.float64) / n * Fs

    f, xf = f[:int(n / 2)], xf[:int(n / 2)]
    xf = np.absolute(xf)

    # print(xf.shape)
    # print(xf[:20])
    period = int(1 / f[np.argmax(xf)])
    #print (period)
    earthquakeperiod.append(period)

#np.savetxt('earthquakeperiod.txt', earthquakeperiod, delimiter=',', fmt='%d')


for filenum1 in range(1, 51):
    filename1 = '/Users/zhengezhao/Desktop/ArizonaPhDstudy/server/static/data/earthquakes/dataforEQ{0}/dataForSIM{0}'.format(str(filenum1)) + '/1_ShearNormalized.txt'
    sim1file = '/Users/zhengezhao/Desktop/ArizonaPhDstudy/git_zhenge/earthquake/shear/probabilityProductKernels/segmentsinnerproduct/segmentssimilarityforEQ{0}EQ{1}'.format(str(filenum1), str(filenum1)) + '.txt'
    sim1 = np.genfromtxt(sim1file, delimiter=',')
    period1 = 13 * earthquakeperiod[filenum1 - 1]
    data1 = np.genfromtxt(filename1)
    data1 = data1.reshape(-1)
    if (len(data1) % period1) != 0:
        segments1 = np.reshape(data1[:-(len(data1) % period1)], [-1, period1])
    else:
        segments1 = np.reshape(data1, [-1, period1])
    for filenum2 in range(filenum1, 51):
        sim2file = '/Users/zhengezhao/Desktop/ArizonaPhDstudy/git_zhenge/earthquake/shear/probabilityProductKernels/segmentsinnerproduct/segmentssimilarityforEQ{0}EQ{1}'.format(str(filenum2), str(filenum2)) + '.txt'
        sim2 = np.genfromtxt(sim2file, delimiter=',')
        sim12file = '/Users/zhengezhao/Desktop/ArizonaPhDstudy/git_zhenge/earthquake/shear/probabilityProductKernels/segmentsinnerproduct/segmentssimilarityforEQ{0}EQ{1}'.format(str(filenum1), str(filenum2)) + '.txt'
        sim12 = np.genfromtxt(sim12file, delimiter=',')
        filename2 = '/Users/zhengezhao/Desktop/ArizonaPhDstudy/server/static/data/earthquakes/dataforEQ{0}/dataForSIM{0}'.format(str(filenum2)) + '/1_ShearNormalized.txt'
        period2 = 13 * earthquakeperiod[filenum2 - 1]
        mperiod = max(period1, period2)
        data2 = np.genfromtxt(filename2)
        data2 = data2.reshape(-1)
        if (len(data2) % period2) != 0:
            segments2 = np.reshape(data2[:-(len(data2) % period2)], [-1, period2])
        else:
            segments2 = np.reshape(data2, [-1, period2])
        ussegments1 = np.ones(shape=(segments1.shape[0], mperiod))
        for i in range(segments1.shape[0]):
            ussegments1[i] = np.interp(np.linspace(0, 1, mperiod), np.linspace(0, 1, period1), segments1[i])
        ussegments2 = np.ones(shape=(segments2.shape[0], mperiod))
        for i in range(segments2.shape[0]):
            ussegments2[i] = np.interp(np.linspace(0, 1, mperiod), np.linspace(0, 1, period2), segments2[i])

        d = DatasetDistance(ussegments1, ussegments2, sim1, sim2, sim12)
        #print (d,period1,period2,segments1.shape,ussegments1.shape,segments2.shape,ussegments2.shape)
        distance[filenum1 - 1, filenum2 - 1] = d
        distance[filenum2 - 1, filenum1 - 1] = d

        #print (sim.shape,period1,period2,segments1.shape,ussegments1.shape,segments2.shape,ussegments2.shape)

np.savetxt('distanceforequakesusedkernel.txt', distance, delimiter=',', fmt='%1.4f')
#print (d.shape)
# similarity matrix


# fig = plt.figure(figsize=(8,8))
# plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0,hspace=0,wspace=0)


# for i,v1 in enumerate(segments):

# 	v1 = v1[:len(v1)-len(v1)%13]
# 	v1 = v1.reshape([-1,13])

# 	v1 = np.sum(v1, axis=1)/13

# 	ax = fig.add_subplot(segments.shape[0]+1,segments.shape[0]+1,i+2)
# 	ax.plot(v1[::40])
# 	ax.set_xticks([])
# 	ax.set_yticks([])
# 	ax = fig.add_subplot(segments.shape[0]+1,segments.shape[0]+1,(i+1)*(segments.shape[0]+1)+1)
# 	ax.plot(v1[::40])
# 	ax.set_xticks([])
# 	ax.set_yticks([])


# ax = plt.subplot2grid([segments.shape[0]+1,segments.shape[0]+1],[1,1],rowspan=segments.shape[0],colspan=segments.shape[0])
# ax.matshow(1-sim, cmap=plt.cm.gray)
# ax.set_xticks([])
# ax.set_yticks([])

# #plt.savefig('foo.png',dpi=330)

# plt.show()
