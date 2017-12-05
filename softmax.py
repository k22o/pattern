#!/usr/bin/env python
# -*- coding: utf-8 -*-

##宣言部
import os
import numpy as np
import scipy as sp
import pickle
import gzip
from numpy.random import *
import random

##変数定義
pix_w = 28 #横のピクセル数
pix_h = 28 #縦のピクセル数
img_size = pix_w * pix_h
unit1_num = img_size #第1階層(入力)のユニット数
unit2_num = 10 #第2階層のユニット数
eta = 0.05 #0.05 #学習係数
threshold = 0 #閾値
batch = 1 #ミニバッチの値
path_train_data = "/home/mech-user/work/roboint/mnist/train-images-idx3-ubyte.gz"
path_train_label = "/home/mech-user/work/roboint/mnist/train-labels-idx1-ubyte.gz"
path_test_data ="/home/mech-user/work/roboint/mnist/t10k-images-idx3-ubyte.gz"
path_test_label ="/home/mech-user/work/roboint/mnist/t10k-labels-idx1-ubyte.gz"


#データセットの解凍
#http://yann.lecun.com/exdb/mnist/
#https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/dataset/mnist.py
def unpickle_label(file_path):
    with gzip.open(file_path,'rb') as f:
        labels = np.frombuffer(f.read(),np.uint8,offset=8)
    return labels
def unpickle_img(file_path):
    with gzip.open(file_path,'rb') as f:
	       data = np.frombuffer(f.read(),np.uint8,offset=16)
    data = data.reshape(-1,pix_w*pix_h)#0-1の値に変更
    return data/255.0

#ソフトマックス関数
#http://kakts-tec.hatenablog.com/entry/2017/02/02/005027
def softmax(x):
    c = np.max(x)# 入力値の中で最大値を取得
    # オーバーフロー対策として、最大値cを引く。こうすることで値が小さくなる
    exp_x = np.exp(x - c);
    sum_exp_x = np.sum(exp_x,axis=0)
    return exp_x / sum_exp_x

##BPの実装(準備)
def update_delta_last(output_calc, output):
    return  (output_calc - output)/10

#出力を求め,フｨードバックする計算部分
def calculation(x1,w12,y,k):

    add = np.ones((1,batch))
    x1 = np.vstack((add,x1)) #バイアス項を付け足す
    w12[:,0]=-1*threshold
    u2 = np.dot(w12,x1)
    x2 = softmax(u2)

    #バックプロパゲーション
    delta2 = update_delta_last(x2, y)
    w12 = w12 - eta/k * np.dot(delta2,x1.T)/batch
    dif = np.absolute(np.linalg.norm(y-x2))
    
    return w12,dif


#学習完了後、推論をして、どこに分類されたかを返す関数
def prediction(x1,w12):
    x1 = np.insert(x1,0,1) #バイアス項を付け足す
    u2 = np.dot(w12,x1)
    x2 = softmax(u2)
    ans = np.argmax(x2)
    return (x2,ans)

#重みの初期設定
def create_mat(unit_a,unit_b):
    new_w = np.random.randn(unit_a,unit_b)
    old_w = np.zeros((unit_a,unit_b))
    return new_w,old_w

#データセットを解凍する
def load_data(path_data,path_label):
    real_data = unpickle_img(path_data)
    label = unpickle_label(path_label)
    return real_data,label

##main
if __name__ == '__main__':

    #####訓練#################################################################################
    #wの初期設定
    w12,old_w12 = create_mat(unit2_num,unit1_num+1)
    #データの読み込み
    real_train_data,train_label = load_data(path_train_data,path_train_label)
    unit_matrix = np.eye(10)

    cnt = 0
    sub = 100
    old_dif=0
    dif = 100
    while(dif > 0.000001 and cnt < 10):
        cnt += 1

        index_data = range(len(real_train_data))
        random.shuffle(index_data)
        real_train_data_copy = np.zeros((len(real_train_data),img_size))
        train_label_copy = np.zeros(len(real_train_data))

        #ミニバッチ用として、ランダムにデータを選ぶ
        for num in range(len(real_train_data)):
            a = index_data[num]
            real_train_data_copy[num,:] = real_train_data[a,:]
            train_label_copy[num] = train_label[a]
        #ソフトマックス回帰の計算
        for i in range(len(real_train_data_copy)/batch):
            x1 = real_train_data_copy[i*batch:(1+i)*batch,:]
            x1 = x1.T
            y = np.zeros((unit2_num,batch))
            for j in range(batch):
                b = int(train_label_copy[i*batch + j])
                y[:,j]=unit_matrix[:,b]
            w12,dif = calculation(x1,w12,y,1)

        print("誤差 {},回数 {}".format(dif,cnt))
        old_dif = dif

    #####テスト#################################################################################

    acc = 0
    all_files = 0
    real_test_data,test_label = load_data(path_test_data,path_test_label)

    for i in range(len(real_test_data)):
        x1 = real_test_data[i,:]
        x2,ans = prediction(x1,w12)
        all_files +=1.0
        if ans == test_label[i]:
            acc += 1.0

    acc = acc * 100 /all_files
    print acc
