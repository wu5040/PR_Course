import matplotlib.pyplot as plt
import numpy as np
#分别存放所有点的横坐标和纵坐标，一一对应
# data = [1,2,3,3.5,6,5]
# y_list = [1,3,4,5,4,7]
# data = data[data[:,2].argsort()]
# #创建图并命名
# plt.figure('Line fig')
# ax = plt.gca()
# #设置x轴、y轴名称
# ax.set_xlabel('x')
# ax.set_ylabel('y')

# #画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
# #参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
# ax.plot(x_list, y_list, color='r', linewidth=1, alpha=0.6)

# plt.show()
# list1 = [[1, 3, 2], [3, 5, 4],[2,3,4]]
# array = np.array(list1)
# array=array[array[:,0].argsort()]
# print(array)
file=open(r"MALE.txt")
file_data1 = np.loadtxt(file, delimiter='\t')
