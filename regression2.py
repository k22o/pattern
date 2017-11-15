# -*- coding:utf-8 -*-
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

####データの読み込み・計算####
'''
もともとのデータセットの中身
 1. mpg:           continuous
 2. cylinders:     multi-valued discrete
 3. displacement:  continuous
 4. horsepower:    continuous
 5. weight:        continuous
 6. acceleration:  continuous
 7. model year:    multi-valued discrete
 8. origin:        multi-valued discrete
 9. car name:      string (unique for each instance)
'''
dataset = np.loadtxt("auto-mpg.csv",delimiter=",",usecols=(0,3,4,2,5))
x_data = np.array([dataset[:,1],dataset[:,2],dataset[:,3],dataset[:,4]])
w0 = np.ones((1,len(dataset)))
x_T = np.r_[w0,x_data]
x = x_T.T
t = np.array(dataset[:,0])
w = (np.linalg.inv( np.dot(x_T,x)))
w = np.dot(w,x_T)
w = np.dot(w,t)

####グラフの描写####
fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel = ("horsepower")
ax.set_ylabel = ("weight")
ax.set_zlabel = ("mpg")
xmin = np.min(dataset[:,1])
xmax = np.max(dataset[:,1])
ymin = np.min(dataset[:,2])
ymax = np.max(dataset[:,2])
ax.scatter(dataset[:,1],dataset[:,2],dataset[:,0])
predict_x = np.arange(xmin,xmax, (xmax-xmin)/30)
predict_y = np.arange(ymin,ymax, (ymax-ymin)/30)
plot_x,plot_y = np.meshgrid(predict_x,predict_y)
plot_z = w[0]*1 + w[1]*plot_x + w[2]*plot_y
ax.plot_wireframe(plot_x,plot_y,plot_z)
plt.show()
