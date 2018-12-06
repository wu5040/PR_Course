import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_data1=np.loadtxt(r'/home/wsg/111aaa/PR_Course/expData/genderdata/boy.txt',delimiter='\t')
file_data2=np.loadtxt(r'/home/wsg/111aaa/PR_Course/expData/genderdata/girl.txt',delimiter='\t')

ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程

for i in file_data1:
    ax.scatter(i[0],i[1],i[2],c='b')

for i in file_data2:
    ax.scatter(i[0],i[1],i[2],c='r')

ax.set_zlabel('height')  # 坐标轴
ax.set_ylabel('weight')
ax.set_xlabel('shoe size')

plt.show()
