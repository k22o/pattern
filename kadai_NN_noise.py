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
import matplotlib.pyplot as plt

##変数定義
pix_w = 28 #横のピクセル数
pix_h = 28 #縦のピクセル数
img_size = pix_w * pix_h
unit1_num = img_size #第1階層(入力)のユニット数
unit2_num = 100 #第2階層のユニット数
unit3_num = 50 #第3階層のユニット数
unit4_num = 10 #第4階層のユニット数=分類数
eta = 0.05#学習係数
threshold = 0.1 #閾値
gain = 4 #シグモイド関数のゲイン
batch = 100 #ミニバッチの値
path_train_data = "mnist/train-images-idx3-ubyte.gz"
path_train_label = "mnist/train-labels-idx1-ubyte.gz"
path_test_data ="mnist/t10k-images-idx3-ubyte.gz"
path_test_label ="mnist/t10k-labels-idx1-ubyte.gz"


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
    data = data.reshape(-1,pix_w*pix_h)
    return data

#データセットをまとめて解凍する
def load_data(path_data,path_label,noise):
    original_data = unpickle_img(path_data)#もともとのピクセルデータ
    label = unpickle_label(path_label)#ラベル用のデータ
    real_data = gray0to1(original_data,noise)#標準化,ノイズの導入
    return real_data,label

#各データ毎の標準化
def normalization(data,idx):
    if idx == 0:#行方向
        mean = data.mean(axis=0)
        mean = mean[np.newaxis,:]
        std =np.std(data,axis=0)
        std = std[np.newaxis,:]
        return (data -mean)/std
    elif idx == 1:#列方向
        mean = data.mean(axis=1)
        mean = mean[:,np.newaxis]
        std =np.std(data,axis=1)
        std = std[:,np.newaxis]
        return (data -mean)/std

#dデータを0-1の値にして必要に応じてノイズいれる
def gray0to1(data,noise):
    ans_data = data /255.0
    ans_data = normalization(ans_data,1)

    ###ノイズ発生箇所
    if noise !=0:
        '''
        #全てに対してランダム　おおもと
        for i in range(len(ans_data)):
            rand_item = randint(0,img_size,img_size*noise/100)
            for j in (rand_item):
               ans_data[i,j] = rand()
        '''
        '''
        #1で固定　パターン1
        for i in range(len(ans_data)):
            rand_item = randint(0,img_size,img_size*noise/100)
            for j in (rand_item):
               ans_data[i,j] = 1
        '''
        '''
        #パターン2
        if noise == 5:
            b = 1
        elif noise == 10:
            b= 3
        elif noise == 15:
            b= 4
        elif noise == 20:
            b= 6
        elif noise == 25:
            b= 7
        else:
            b = 0

        for i in range(len(ans_data)):
            rand_item = randint(0,28,b)
            for j in (rand_item):
                for k in range(28):
                    ans_data[i,j + k*28] = 1.0
        '''

        #パターン3
        rand_item = randint(5,19,1)
        rand_item = [10]
        if noise == 5:
            b = 1
        elif noise == 10:
            b= 3
        elif noise == 15:
            b= 4
        elif noise == 20:
            b= 6
        elif noise == 25:
            b= 7
        else:
            b = 1

        for i in range(28):
            ans_data[:,(rand_item[0] + i*28):(rand_item[0] + i*28 + b)] = 1.0

    return ans_data


##活性化関数の定義　参考文献(4),(10)より引用
def sigmoid_pre(x):
    sigmoid_range = 34.538776394910684

    if x*gain <= -sigmoid_range:
        return 1e-15
    if x*gain >= sigmoid_range:
        return 1.0 - 1e-15

    return 1.0 / (1.0 + np.exp(-1.0*x*gain))

#http://kakts-tec.hatenablog.com/entry/2017/02/02/005027
def softmax(x):
    c = np.max(x)# 入力値の中で最大値を取得
    # オーバーフロー対策として、最大値cを引く。こうすることで値が小さくなる
    exp_x = np.exp(x - c);
    sum_exp_x = np.sum(exp_x,axis=0)
    return exp_x * np.power(sum_exp_x,-1)


##BPの実装
####参考文献(5)http://kaisk.hatenadiary.com/entry/2014/09/20/185553
#最終層の重み-softmax
def update_delta_last(output_calc, output):
    return  (output_calc - output)
#それ以外-sigmoid
def update_delta(w_next,delta_next,x):
    return np.dot(w_next.T, delta_next) * x * (1- x)* gain

#出力を求め,フｨードバックする計算部分。
def calculation(x1,w12,w23,w34,y,k):

    add = np.ones((1,batch))
    x1 = np.vstack((add,x1)) #バイアス項を付け足す
    w12[:,0]=-1*threshold
    u2 = np.dot(w12,x1)
    x2 = sigmoid(u2)

    x2[0,:]=1
    w23[:,0]=-1*threshold
    u3 = np.dot(w23,x2)
    x3 = sigmoid(u3)

    x3[0,:]=1;
    w23[:,0]=-1*threshold
    u4 = np.dot(w34,x3)
    x4 = softmax(u4)

    #バックプロパゲーション
    #kにループ回数を指定すると、学習効率の時間減衰、1だと減衰なし
    delta4 = update_delta_last(x4, y)
    w34 = w34 - eta / k * np.dot(delta4,x3.T)/batch
    delta3 = update_delta(w34,delta4,x3)
    w23 = w23 - eta / k * np.dot(delta3,x2.T)/batch
    delta2 = update_delta(w23,delta3,x2)
    w12 = w12 - eta / k * np.dot(delta2,x1.T)/batch

    dif = np.absolute(np.linalg.norm(y-x4))#誤差

    return (w12,w23,w34,x4,dif)


#学習完了後、推論をして、どこに分類されたかを返す関数
def prediction(x1,w12,w23,w34):
    x1 = np.insert(x1,0,1) #バイアス項を付け足す
    w12[:,0]=-1*threshold
    u2 = np.dot(w12,x1)
    x2 = sigmoid(u2)
    x2[0]=1
    w23[:,0]=-1*threshold
    u3 = np.dot(w23,x2)
    x3 = sigmoid(u3)
    x3[0]=1
    w23[:,0]=-1*threshold
    u4 = np.dot(w34,x3)
    x4 = softmax(u4)
    ans = np.argmax(x4)
    return (x4,ans)

#重みの初期設定
def create_mat(unit_a,unit_b):
    new_w = np.random.randn(unit_a,unit_b)/np.sqrt(unit_b -1)
    #old_w = np.zeros((unit_a,unit_b))#wの前の回との比較の際には用いる
    return new_w #,old_w


##main（ノイズを加味しないとき。加味するときはマークダウンo）
if __name__ == '__main__':

    #####訓練#################################################################################
    '''
    w12,old_w12 = create_mat(unit2_num+1,unit1_num+1)
    w23,old_w23 = create_mat(unit3_num+1,unit2_num+1)
    w34,old_w34 = create_mat(unit4_num,unit3_num+1)
    '''

    #シグモイド関数を行列に適応できるように拡張
    sigmoid = np.vectorize(sigmoid_pre)

    #重みの初期値の設定
    w12 = create_mat(unit2_num+1,unit1_num+1)
    w23 = create_mat(unit3_num+1,unit2_num+1)
    w34 = create_mat(unit4_num,unit3_num+1)

    noise_x = np.zeros((5))
    noise_y = np.zeros((5))

    for noise in range(5):

        real_train_data,train_label = load_data(path_train_data,path_train_label,0)
        unit_matrix = np.eye(10)

        #length = len(real_train_data)
        length = 20000#使用するデータ

        cnt = 0
        dif = 100
        old_dif = 0
        while(dif > 0.02 and cnt < 15):
            cnt += 1

            #ミニバッチ用の行列の準備
            index_data = range(length)
            random.shuffle(index_data)
            real_train_data_copy = np.zeros((length,img_size))
            train_label_copy = np.zeros(length)

            #ミニバッチ用として、ランダムにデータを選ぶ
            for num in range(length):
                a = index_data[num]
                real_train_data_copy[num,:] = real_train_data[a,:]
                train_label_copy[num] = train_label[a]

            #NNの計算
            for i in range(length/batch):
                x1 = real_train_data_copy[i*batch:(1+i)*batch,:]
                x1 = x1.T
                y = np.zeros((unit4_num,batch))
                for j in range(batch):
                    b = int(train_label_copy[i*batch + j])
                    y[:,j]=unit_matrix[:,b]
                w12,w23,w34,x4,dif = calculation(x1,w12,w23,w34,y, cnt/7.0)#7

            #前との誤差の計算
            old_dif = dif
            #print("繰り返し：{} dif:{}".format(cnt,dif))

    #####テスト#################################################################################

        acc = 0
        all_files = 0
        real_test_data,test_label = load_data(path_test_data,path_test_label,(noise+1)*5)

        for i in range(5000):
            x1 = real_test_data[i,:]
            x4,ans = prediction(x1,w12,w23,w34)
            all_files +=1.0
            if ans == test_label[i]:
                acc += 1.0

        acc = acc * 100 /all_files
        print ("accuracy:{}".format(acc))

        noise_x[noise]=(noise+1)*5
        noise_y[noise]=acc

    plt.title("the retationship between noise and accuracy")
    plt.xlabel("noise ratio")
    plt.ylabel("accuracy")
    plt.plot(noise_x,noise_y,color="red",marker="o")
    plt.show()
