# -*- coding:utf-8 -*-
import numpy as np
from numpy.random import *
import scipy as sp
from matplotlib import pyplot as plt


####配列内の各要素との距離を算出して配列にする関数
def distance_list(target,lst):
        distance = np.array([])
        for i in range (len(lst)):
            distance = np.append(distance,np.linalg.norm(target-lst[i]))
        return distance

####配列の中から最小値(の番目)を探す関数
def min_list(lst):
    min_value = 1000
    min_num = 100
    for i in range(len(lst)):
        if min_value > lst[i] :
                min_value = lst[i]
                min_num = i
    return min_num


########以下、アルゴリズムの流れ########
########データを読み込む/クラス分け用の配列を作る(現在のクラス・1回前のクラスの値)(0〜k-1まで)
dataset = np.loadtxt("iris_dataset.csv",delimiter=",",dtype="double")
class_list = np.empty((0,3) ,float)
for i in range (len(dataset)):
        class_list = np.append(class_list,np.array([[i,100,100]]),axis=0)
fig =plt.figure()
for k in range(2,6):
        ####重心を仮決めする
        rand_list = randint(0,len(dataset),size=k)
        center = np.array([dataset[rand_list[i]] for i in range(k)])

        ####安定するまでループさせる
        while True:
                ##一番距離の近いものを選んでその番号をclass_listにセットする(クラス分け)
                for i in range(len(dataset)):
                        class_list[i][2] = class_list[i][1]
                        class_list[i][1] = min_list(distance_list(dataset[i],center))

                ##重心を算出する
                for i in range (k):
                        sum = np.array([0.0,0.0,0.0,0.0])
                        counter = 0
                        for j in range (len(dataset)):
                                if class_list[j][1] == i:
                                        counter += 1
                                        sum += dataset[j]
                        center[i] = sum /counter

                ##クラス分けの変化の度合いを調べる
                same_counter = 0
                for i in range(len(class_list)):
                        if  class_list[i][1] == class_list[i][2]:
                                same_counter += 1
                if same_counter > len(class_list) * 0.99:
                        break


########プロットする

        plt.subplot(2,2,k-1)
        colors = ['red','blue','yellow','green','pink']

        xc = [0]*k
        yc = [0]*k
        for i in range(k):
                xc[i]= center[i][3]
                yc[i]= center[i][2]

        for i in range(k):
                plot_data = np.empty((0,2),float)
                for j in range(len(class_list)):
                        if class_list[j][1] == i:
                                plot_data = np.append(plot_data,np.array([[dataset[j][3],dataset[j][2]]]),axis=0)
                plt.scatter(plot_data[:,0],plot_data[:,1], color = colors[i])

        plt.scatter(xc,yc, c='black',marker='x')
        plt.xlabel("petal width")#配列の4番目
        plt.ylabel("petal length")#配列の3番目
plt.show()
