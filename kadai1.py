# -*- coding:utf-8 -*-

import numpy as np
import scipy as sp
import csv
from matplotlib import pyplot as plt

############### クラス判断関数  ###############
def detect(setosa,versicolor,virginica):
    max_value = max(setosa,versicolor,virginica)
    if max_value == setosa:
        return "setosa"
    elif max_value == versicolor:
        return "versicolor"
    else:
        return "virginica"
    
########配列などの下準備 ########
#データセットを格納する配列
dataset = np.loadtxt("iris_dataset.csv",delimiter=",",dtype="double")#特徴量を記述したデータ

iris_name =[0]*len(dataset)
with open("iris_name.csv","rb") as f:#分類名が記されたデータ
    reader = csv.reader(f)
    iris_name =[e for e in reader]

counter_iris = [0]*3 #どの種類のアヤメに分類されたかカウントする
k_result_list = []   #正答率を記録するリスト


########kの値に応じて計算########
for k in range(1,31):
    counter_result = 0
    
    #########1つずつ除いてKNN########
    for i in range(len(dataset)):
        counter_iris = [0,0,0]#setosa,versicolor,cirginica

        ####ノルムを求めてnormリストに格納####
        norm=[]
        for j in range(len(dataset)):
            norm.append([np.linalg.norm(dataset[j]-dataset[i]),iris_name[j][0]])#ノルム
            ####ソートして距離の短いk個を考える。dataset[i]は0なので除く####
        list.sort(norm)

        for l in range(1,k+1):
            if norm[l][1]=="setosa":
                counter_iris[0] += 1
            elif norm[l][1]=="versicolor":
                counter_iris[1] += 1
            else:
                counter_iris[2] += 1
       
        ####3種類のうち、どれが一番多いか調べ、それが本来の値と同じか比較####
        if detect(counter_iris[0],counter_iris[1],counter_iris[2]) == iris_name[i][0]:
            counter_result += 1
        
    #####正答率の算出/結果リストへの格納####
    percent_correct = 100.0 * counter_result / len(dataset)
    k_result_list.append(percent_correct)    
    
    
####グラフをかく####
x = [0] * 30
for i in range(0,30):
    x[i] = i+1
    
plt.plot(x,k_result_list,marker='o')
plt.title("k Nearest Neighbor Algorithm")
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()

