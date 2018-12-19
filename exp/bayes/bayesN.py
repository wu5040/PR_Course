import numpy as np
from math import sqrt
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def loadDataSet(intX, intY):
    file_data1 = np.loadtxt(
        r'C:/111aaa/PR_Course/expData/genderdata/boy.txt', delimiter='\t')
    file_data2 = np.loadtxt(
        r'C:/111aaa/PR_Course/expData/genderdata/girl.txt', delimiter='\t')

    lenData1 = len(file_data1)
    groupListMale = np.empty([lenData1, 2], dtype=float)
    classVecMale = np.ones(lenData1)
    for i in range(lenData1):
        groupListMale[i][0] = file_data1[i][intX]
        groupListMale[i][1] = file_data1[i][intY]

    lenData2 = len(file_data2)
    groupListFemale = np.empty([lenData2, 2], dtype=float)
    classVecFemale = np.zeros(lenData2)
    for i in range(lenData2):
        groupListFemale[i][0] = file_data2[i][intX]
        groupListFemale[i][1] = file_data2[i][intY]

    return lenData1, groupListMale, classVecMale, lenData2, groupListFemale, classVecFemale


def GaussianFun(ListX, lenX):
    # mu_sum=np.zeros(2)
    # for item in ListX:
    #     mu_sum+=item
    # mu=mu_sum/lenX
    # mu.shape=(1,2)

    mu = np.mean(ListX, axis=0)

    sigma_sum = 0
    for item in ListX:
        x = item-mu
        x.shape = (1, 2)
        sigma_sum += np.dot(x.T, x)
    sigma = sigma_sum/lenX

    # sigma=np.cov(ListX.T)
    return mu, sigma


def build_gaussian_layer(mu, sigma):
    x = np.arange(130, 200, 1)
    y = np.arange(30, 100, 1)
    x, y = np.meshgrid(x, y)
    z = np.exp(-((y-mu)**2 + (x - mu)**2)/(2*(sigma**2)))
    z = z/(np.sqrt(2*np.pi)*sigma)
    return (x, y, z)


def getSigma(lenData, listX, mu):
    sigma_sum = np.zeros(2)
    for item in listX:
        sigma_sum += (item-mu)**2
    sigma = sigma_sum/lenData
    for i in range(2):
        sigma[i] = sqrt(sigma[i])
    return sigma

# normfun正态分布函数


def normFun(x, y, muX, sigmaX, muY, sigmaY):
    pdf = (np.exp(-((x - muX)**2) / (2 * sigmaX**2)) / (sigmaX * np.sqrt(2*np.pi))) * \
        (np.exp(-((y - muY)**2) / (2 * sigmaY**2)) / (sigmaY * np.sqrt(2*np.pi)))
    return pdf


lenData1, groupListMale, classVecMale, lenData2, groupListFemale, classVecFemale = loadDataSet(
    0, 1)
mu1, sigma1 = GaussianFun(groupListMale, lenData1)
mu2, sigma2 = GaussianFun(groupListFemale, lenData2)

print("男生方差\n",sigma1)
print("女生方差\n",sigma2)

sigma1 = getSigma(lenData1, groupListMale, mu1)
print("男生均值", mu1)
print("男生标准差", sigma1)

sigma2 = getSigma(lenData2, groupListFemale, mu2)
print("女生均值", mu2)
print("女生标准差", sigma2)

x = np.arange(140, 200, 1)
y = np.arange(30, 90, 1)
x, y = np.meshgrid(x, y)
z1 = normFun(x, y, mu1[0], sigma1[0], mu1[1], sigma1[1])
z2 = normFun(x, y, mu2[0], sigma2[0], mu2[1], sigma2[1])

fig = plt.figure()
ax = Axes3D(fig)

ax.plot_surface(x, y, z1, cmap="rainbow")
ax.plot_surface(x, y, z2, cmap="rainbow")

ax.set_zlabel('Probability')  # 坐标轴
ax.set_ylabel('weight')
ax.set_xlabel('height')

plt.show()
