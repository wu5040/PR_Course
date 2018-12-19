import numpy as np
from math import sqrt, log
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def loadDataSet(intX, intY):
    file_data1 = np.loadtxt(r'C:\111aaa\PR_Course\expData\genderdata\MALE.txt', delimiter='\t')

    file_data2 = np.loadtxt(r'C:\111aaa\PR_Course\expData\genderdata\FEMALE.txt', delimiter='\t')

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
def normFun(x, y, mu, sigma):
    pdf = (np.exp(-((x - mu[0])**2) / (2 * sigma[0]**2)) / (sigma[0] * np.sqrt(2*np.pi))) * \
        (np.exp(-((y - mu[1])**2) / (2 * sigma[1]**2)) /
         (sigma[1] * np.sqrt(2*np.pi)))
    return pdf


def classifyNB(x, y):
    lenData1, groupListMale, classVecMale, lenData2, groupListFemale, classVecFemale = loadDataSet(
        0, 1)
    
    #先验概率
    priorP1=lenData1/(lenData1+lenData2)
    priorP2=lenData2/(lenData1+lenData2)

    #计算 mu,sigma
    mu1, sigma1 = GaussianFun(groupListMale, lenData1)
    mu2, sigma2 = GaussianFun(groupListFemale, lenData2)
    sigma1 = getSigma(lenData1, groupListMale, mu1)
    sigma2 = getSigma(lenData2, groupListFemale, mu2)

    #判别函数
    gXY = log(normFun(x, y, mu1, sigma1)) + log(priorP1) - \
        log(normFun(x, y, mu2, sigma2)) - log(priorP2)
    if gXY >= 0:
        return 1
    else:
        return 0


def LOO(x,y):
    '''
        留一法
        返回TP,FN,FP,TN
    '''
    lenData1, groupListMale, classVecMale, lenData2, groupListFemale, classVecFemale = loadDataSet(
        x, y)


    #先验概率
    priorP1=(lenData1-1)/(lenData1+lenData2-1)
    priorP2=lenData2/(lenData1+lenData2-1)

    mu2, sigma2 = GaussianFun(groupListFemale, lenData2)
    sigma2 = getSigma(lenData2, groupListFemale, mu2)
    errMale = 0
    resArr1 = []
    for i in range(lenData1):
        x = groupListMale[i][0]
        y = groupListMale[i][1]
        group = np.delete(groupListMale, i, axis=0)
        mu1, sigma1 = GaussianFun(group, lenData1-1)
        sigma1 = getSigma(lenData1-1, group, mu1)

        gXY = log(normFun(x, y, mu1, sigma1)) + log(priorP1) - \
            log(normFun(x, y, mu2, sigma2)) - log(priorP2)

        if gXY < 0:  # 判断错误
            errMale = errMale+1
        resArr1.append([gXY, 1])


    #先验概率
    priorP1=lenData1/(lenData1+lenData2-1)
    priorP2=(lenData2-1)/(lenData1+lenData2-1)

    mu1, sigma1 = GaussianFun(groupListMale, lenData1)
    sigma1 = getSigma(lenData1, groupListMale, mu1)
    errFemale = 0
    for i in range(lenData2):
        x = groupListFemale[i][0]
        y = groupListFemale[i][1]
        group = np.delete(groupListFemale, i, axis=0)

        mu2, sigma2 = GaussianFun(group, lenData2-1)
        sigma2 = getSigma(lenData2-1, group, mu2)

        gXY = log(normFun(x, y, mu1, sigma1)) + log(priorP1) - \
            log(normFun(x, y, mu2, sigma2)) - log(priorP2)
            
        if gXY >= 0:  # 判断错误
            errFemale = errFemale+1
        resArr1.append([gXY, 0])

    # 返回TP,FN,FP,TN
    return resArr1,lenData1-errMale, errMale, errFemale, lenData2-errFemale


if __name__ == "__main__":
    lenData1, groupListMale, classVecMale, lenData2, groupListFemale, classVecFemale = loadDataSet(
        0, 1)
    mu1, sigma1 = GaussianFun(groupListMale, lenData1)
    mu2, sigma2 = GaussianFun(groupListFemale, lenData2)

    sigma1 = getSigma(lenData1, groupListMale, mu1)
    print("男生均值", mu1)
    print("男生标准差", sigma1)

    sigma2 = getSigma(lenData2, groupListFemale, mu2)
    print("女生均值", mu2)
    print("女生标准差", sigma2)

    x = np.arange(140, 200, 1)
    y = np.arange(30, 90, 1)
    x, y = np.meshgrid(x, y)
    z1 = normFun(x, y, mu1, sigma1)
    z2 = normFun(x, y, mu2, sigma2)

    print(type(z1))
    
    

    # fig1 = plt.figure()
    # ax1 = Axes3D(fig1)

    # ax1.plot_surface(x, y, z1, cmap="rainbow")
    # ax1.plot_surface(x, y, z2, cmap="rainbow")

    # ax1.set_zlabel('Probability')  # 坐标轴
    # ax1.set_ylabel('weight')
    # ax1.set_xlabel('height')

    # fig2 = plt.figure()
    # ax2 = Axes3D(fig2)

    # #先验概率
    # priorP1=lenData1/(lenData1+lenData2)
    # priorP2=lenData2/(lenData1+lenData2)
    
    # z = z1*priorP1+z2*priorP2

    # ax2.plot_surface(x, y, (z1*priorP1)/z, cmap="rainbow")
    # ax2.plot_surface(x, y, (z2*priorP2)/z, cmap="rainbow")
    # ax2.set_zlabel('Probability')  # 坐标轴
    # ax2.set_ylabel('weight')
    # ax2.set_xlabel('height')

    # plt.show()
