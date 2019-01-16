#-*-coding:utf-8-*-
from numpy import *
import operator
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import operator
from mpl_toolkits.mplot3d import Axes3D
import random



def autoNorm(dataMat):
    #归一化
    minVals=dataMat.min(0) #最小值存到minVals里
    maxVals=dataMat.max(0) #最大值存到maxVals里
    ranges=maxVals-minVals #每列最大最小值之差,存到ranges里
    newDataMat=zeros(shape(dataMat)) #用来存归一化后的矩阵
    m=dataMat.shape[0] #取第0维即行数
    newDataMat=dataMat-tile(minVals,(m,1)) #把最小值重复m成m行,用原值减去
    newDataMat=newDataMat/tile(ranges,(m,1)) #把减完的每个偏差值除以自己列最大和最小的差值
    return newDataMat,ranges,minVals #返回归一化之后的矩阵,原范围,原最小值

#训练集交错混合
def MyFix(dataMat1,dataMat0):
    i=j=0
    len1=len(dataMat1)
    len0=len(dataMat0)
    
    dataMat=[0]*(len1+len0)
    labelVec=[0]*(len1+len0) 
    
    while(i<len1 or j<len0):
        if i<len1:
            dataMat[i+j]=dataMat1[i]
            labelVec[i+j]=1
            i+=1
        if j<len0:
            dataMat[i+j]=dataMat0[j]
            labelVec[i+j]=0
            j+=1
    return dataMat,labelVec

#用长为lnth的rdmLst列表打乱用dataMat,labelVec表征的同样数目的样本集
def MyUpset(dataMat,labelVec,rdmLst,lnth):
    for i in range(lnth): #对随机数表中每一项条目所表示的下标
        #做交换以打乱样本集
        dataMat[i],dataMat[rdmLst[i]]=dataMat[rdmLst[i]],dataMat[i]
        labelVec[i],labelVec[rdmLst[i]]=labelVec[rdmLst[i]],labelVec[i]
    return dataMat,labelVec

#对test集中的某个样本向量做测试,返回分类结果
def Tst(tstVec,traMat,traLab):
    traMat=mat(traMat)
    rowNum=traMat.shape[0] #训练集行数
    #用tile()将输入的特征向量重复成和训练集特征向量一样多的行
    #变成2维,里面的维度重复1次,外面一层重复rowNum次
    diffMat=tile(tstVec,(rowNum,1))
    #相减得到偏差矩阵
    diffMat=diffMat-traMat
    #将减出来的偏差矩阵每个元素平方
    sqDiffMat=multiply(diffMat,diffMat)
    #对行求和,表示这个实例和这行对应的训练集实例的L2范数的平方
    sqDistances=sqDiffMat.sum(axis=1)
    ###print(sqDistances)
    #mat变列表
    sqDistances=[sqDistances[i,0] for i in range(len(sqDistances))]
    #再变成array对象(为了argsort())
    sqDistances=array(sqDistances)
    #为了方便就不开根号(**0.5)了
    #argsort()返回其从小到大排序的排序索引序列
    sortIndex=sqDistances.argsort()
    #找前k个距离最近的,也就是排序下标从0~k-1的
    Vote1=Vote0=0 #男女投票数都初始化为0
    for i in range(5):
        #第i近(从0计数)训练集实例的标签,男生是1女生是0
        if traLab[sortIndex[i]]==1:
            Vote1+=1
        elif traLab[sortIndex[i]]==0:
            Vote0+=1
    return 1 if Vote1>Vote0 else 0

#找出考试集中考试错误的那些样本的下标
#[修正]传入labelVec以确定死亡样本
#[停用]停用参数tstLab
def FindErr(tstMat,tstLab,traMat,traLab,sword,labelVec):
    whoErr=[]
    #对于考试集中的每个样本(sword就等于len(tst*))
    for i in range(sword):
        #如果有考试资格(非-1,即不是被剔除掉的僵尸样本),且分类错误
        #[修正]tstLab改用labelVec
        if labelVec[i]!=-1 and Tst(tstMat[i],traMat,traLab)!=labelVec[i]:
            whoErr.append(i) #把错误者的下标i记录下来
        ###break
    return whoErr

def Go(a,b):
    #获取训练集
    dataMat1 = loadtxt(r'C:\111aaa\PR_Course\expData\genderdata\MALE.txt', delimiter='\t')
    dataMat0 = loadtxt(r'C:\111aaa\PR_Course\expData\genderdata\FEMALE.txt', delimiter='\t')
    
    #训练集交错混合
    dataMat,labelVec=MyFix(dataMat1,dataMat0)

    #归一化
    dataMat,ranges,minVals=autoNorm(mat(dataMat))
   
    #选取特征
    dataMat=dataMat.tolist()
    dataMat=[[dataMat[r][a],dataMat[r][b]] for r in range(len(dataMat))]
    
    #样本集的长度
    dtmtLnth=len(dataMat)
    
    #划分样本集的比例
    sword=int(dtmtLnth/2)
    
    #剪辑近邻法剪10次
    for page in range(10):

        tstMat=dataMat[0:sword]

        tstLab=labelVec[0:sword]


        if page==0:
            #[修正]显示考试集,传入labelVec以确定死亡样本
            MyPic(tstMat,tstLab,"Clips 0 times",a,b,labelVec)
        traMat=dataMat[sword:]
        traLab=labelVec[sword:]
        #计算并返回考试集中考试错误的样本下标
        #[修正]传入labelVec以确定死亡样本
        whoErr=FindErr(tstMat,tstLab,traMat,traLab,sword,labelVec)
        #[添加]输出检验
        print("分类错误%d次"%len(whoErr))
        #把这些分类错误的样本的特征向量变成-1
        #在调用MyPic显示输出或者FindErr考试时,特判跳过它们即可
        #避免了对列表项的删除(那样很费时)
        for index in whoErr:
            #[原来的错误]tstLab[index]=-1
            #[修正]切片只能切得值,不能切得引用,另需在LabelVec里改
            labelVec[index]=-1
        #[修正]显示考试集,传入labelVec以确定死亡样本
        MyPic(tstMat,tstLab,"Clips %d times"%(page+1),a,b,labelVec)
        ###print(whoErr)
        #[修正]随机数表必须混合训练集和测试集
        rdmLst=[random.randint(0,dtmtLnth-1) for i in range(dtmtLnth)]
        #对刚刚分出来的训练集和测试集混合打乱,[修正]长度
        dataMat,labelVec=MyUpset(dataMat,labelVec,rdmLst,dtmtLnth)
        ###print(dataMat,labelVec)

#[修正]绘图,传入labelVec以确定死亡样本
#[停用]停用参数subLab
def MyPic(subMat,subLab,str,i,j,labelVec):
    manX=[]
    manY=[]
    girlX=[]
    girlY=[]
    if i==0:
        xlab='Height'
    elif i==1:
        xlab='Weight'
    elif i==2:
        xlab='Shoe Size'
    if j==0:
        ylab='Height'
    elif j==1:
        ylab='Weight'
    elif j==2:
        ylab='Show Size'
    
    for r in range(len(subLab)):
        if labelVec[r]==1:
            manX.append(subMat[r][0])
            manY.append(subMat[r][1])
        elif labelVec[r]==0:
            girlX.append(subMat[r][0])
            girlY.append(subMat[r][1])
    #绘制
    plt.figure()
    plt.scatter(manX,manY,c=u'b',marker='.')
    plt.scatter(girlX,girlY,c=u'r',marker='.')
    plt.title(str)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()

Go(0,1)
