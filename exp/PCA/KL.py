import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn import decomposition



class KL:

    # def computeR(self):
    #     # 计算自相关矩阵R
    #     R=0
    #     for i in self.X:
    #         i=np.mat(i)
    #         R+=np.dot(i.T,i)
    #     R=R/len(self.X)
    #     print("自相关矩阵R:\n",R)
    #     return R
        
    def computeCov(self):
        X_cov=np.cov(self.X.T)
        # print("协方差矩阵:\n",X_cov)
        return X_cov
    
    def computeSw(self):
        X0=self.X[self.y==0]
        X1=self.X[self.y==1]

        X0_cov=np.cov(X0.T)
        X1_cov=np.cov(X1.T)

        Sw=X0_cov+X1_cov
        # print("类内散度矩阵:\n",Sw)
        return Sw

    def computeSwSb(self):
        Sw=self.computeSw()
        Sb=self.computeCov()-Sw

        Swb=np.mat(Sw).I*Sb
        # print("类内类间距离:\n",Swb)
        return Swb

    def __init__(self,X,y,whichMat):
        '''
            X---data
            y---label
            whichMat:   0---Sw
                        1---Cov
                        2---Swb
        '''
        self.X=X
        self.y=y
        if whichMat==0:
            mat=self.computeSw()
        elif whichMat==1:
            mat=self.computeCov()
        elif whichMat==2:
            mat=self.computeSwSb()

        # 计算特征值和特征向量
        a,b=np.linalg.eig(mat)
        # print("特征值:\n",a)
        # print("特征向量:\n",b)

        self.LambdaArr=a

        sorted_indices = np.argsort(a)

        index=sorted_indices[:-2-1:-1]
        # print("特征值从大到小排序，下标值为:",index)
        W=b.T[index]
        # print("由特征值对应的特征向量，得到变换矩阵W为:\n",W)

        X_KL=(W*np.mat(self.X).T).T
        self.X_trans=np.asarray(X_KL)
        # print("特征提取的结果:\n",self.X_trans)
    

    def getLambdaArr(self):
        '''
            返回特征值向量
        '''
        return self.LambdaArr

    def transform(self):
        '''
            返回特征特取后X
        '''
        return self.X_trans


if __name__=="__main__":
    file_data = np.loadtxt(
        r'C:\111aaa\PR_Course\expData\studentdataset.csv', delimiter=',')

    # 身高、体重、鞋码
    X = file_data[:,:3]
    # 性别
    y = file_data[:, 3]

    kl1=KL(X,y,0)
    X1=kl1.transform()

    kl2=KL(X,y,1)
    X2=kl2.transform()

    kl3=KL(X,y,2)
    X3=kl3.transform()

    plt.figure()
    y_unique=np.unique(y)
    colors=['r','b']
    for this_y,color in zip(y_unique,colors):
        this_X=X1[y==this_y]
        plt.scatter(this_X[:,0],this_X[:,1],c=color)
    plt.title("Sw")

    plt.figure()
    y_unique=np.unique(y)
    colors=['r','b']
    for this_y,color in zip(y_unique,colors):
        this_X=X2[y==this_y]
        plt.scatter(this_X[:,0],this_X[:,1],c=color)
    plt.title("Cov")


    plt.figure()
    y_unique=np.unique(y)
    colors=['r','b']
    for this_y,color in zip(y_unique,colors):
        this_X=X3[y==this_y]
        plt.scatter(this_X[:,0],this_X[:,1],c=color)
    plt.title("Swb")

    plt.show()

