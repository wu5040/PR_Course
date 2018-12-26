import sys
sys.path.append(r'C:\111aaa\PR_Course\exp')

from bayes.bayesN import loadDataSet, GaussianFun
from bayes.DecisionBoundary import getBoundary
import numpy as np
import matplotlib.pyplot as plt

lenData1, groupListMale, classVecMale, lenData2, groupListFemale, classVecFemale = loadDataSet(
    0, 1)

matMale = np.mat(groupListMale).T
matFemale = np.mat(groupListFemale).T

arrMale = np.array(matMale)
arrFemale = np.array(matFemale)

mu1, sigma1 = GaussianFun(groupListMale, lenData1)
mu2, sigma2 = GaussianFun(groupListFemale, lenData2)

mu1 = np.mat(mu1)
mu2 = np.mat(mu2)
print(mu1, mu2)

# 类内散度矩阵Sw
S1 = 0
S2 = 0
for x in groupListMale:
    x = np.mat(x)
    S1 += ((x-mu1).T)*(x-mu1)

for x in groupListFemale:
    x = np.mat(x)
    S2 += ((x-mu2).T)*(x-mu2)

Sw = S1+S2
print(S1)
print(S2)
print(Sw)

omega_x = Sw.I*((mu1-mu2).T)
omega_o = omega_x.T*((mu1+mu2).T)/2
print(omega_x)
print(mu1+mu2)
print(float(omega_o))

x = np.arange(140, 200, 1)
y = (float(omega_o)-float(omega_x[0])*x)/float(omega_x[1])

x1, y1 = getBoundary()

plt.figure()
plt.plot(x, y, linewidth=2, label='Fisher')
plt.plot(x1, y1, linewidth=2, label='Bayes')
plt.scatter(arrMale[0], arrMale[1], c='b', s=10, marker='v', label='Male')
plt.scatter(arrFemale[0], arrFemale[1], c='r',
            s=10, marker='v', label='Female')
plt.xlim(140, 200)
plt.ylim(30, 90)
plt.legend()
plt.show()
