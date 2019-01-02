import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(intX, intY):
    file_data1 = np.loadtxt(
        r'C:\111aaa\PR_Course\expData\genderdata\MALE.txt', delimiter='\t')

    file_data2 = np.loadtxt(
        r'C:\111aaa\PR_Course\expData\genderdata\FEMALE.txt', delimiter='\t')

    lenData1 = len(file_data1)
    groupListMale = np.empty([lenData1, 3], dtype=float)
    classVecMale = np.ones(lenData1)

    mat=np.mat(file_data1).T
    heightMax=np.max(mat[0])
    weightMax=np.max(mat[1])

    mat=np.mat(file_data2).T
    heightMin=np.min(mat[0])
    weightMin=np.min(mat[1])

    hh=heightMax-heightMin
    ww=weightMax-weightMin

    for i in range(lenData1):
        groupListMale[i][0] = (file_data1[i][intX]-heightMin)/hh
        groupListMale[i][1] = (file_data1[i][intY]-weightMin)/ww
        groupListMale[i][2] = 1

    lenData2 = len(file_data2)
    groupListFemale = np.empty([lenData2, 3], dtype=float)
    classVecFemale = np.zeros(lenData2)
    for i in range(lenData2):
        groupListFemale[i][0] = -(file_data2[i][intX]-heightMin)/hh
        groupListFemale[i][1] = -(file_data2[i][intY]-weightMin)/ww
        groupListFemale[i][2] = -1

    return lenData1, groupListMale, lenData2, groupListFemale


lenData1, groupListMale, lenData2, groupListFemale = loadDataSet(0, 2)

matMale = np.mat(groupListMale).T
matFemale = np.mat(groupListFemale).T

arrMale = np.array(matMale)
arrFemale = np.array(matFemale)

print(groupListMale[0])
print(groupListFemale[0])
# 设初始权向量
omega = np.mat([1.0, 1.0, 1.0])
print(omega)

x_list=[]
y_list=[]

i=0
while(i<1000):
    errSum = 0
    change = 0
    for x in groupListMale:
        xMat = np.mat(x).T
        if omega*xMat <= 0:
            errSum += xMat
            change += 1
    for x in groupListFemale:
        xMat = np.mat(x).T
        if omega*xMat <= 0:
            errSum += xMat
            change += 1

    a=np.array(omega)[0][0]
    b=np.array(omega)[0][1]
    x_list.append(a)
    y_list.append(b)

    # if change<20:
    #     break
    omega += errSum.T
    i+=1

print(omega)
a=np.array(omega)[0][0]
b=np.array(omega)[0][1]
c=np.array(omega)[0][2]

x=np.arange(0,1,0.01)
y=(-a*x-c)/b

plt.figure()
plt.plot(x,y)
plt.scatter(arrMale[0], arrMale[1], c='b', s=10, marker='v', label='Male')
plt.scatter(-arrFemale[0], -arrFemale[1], c='r', s=10, marker='v', label='Female')

plt.figure()
plt.plot(x_list, y_list)

plt.show()
