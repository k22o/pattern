#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import scipy as sp
import os
import pickle
import gzip
from numpy.random import *
import random

pix_w = 28 #横のピクセル数
pix_h = 28 #縦のピクセル数
img_size = pix_w * pix_h
k = 20

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
    data = data.reshape(-1,pix_w*pix_h)
    return data

#データセットをまとめて解凍する
def load_data(path_data,path_label):
    original_data = unpickle_img(path_data)#もともとのピクセルデータ
    label = unpickle_label(path_label)#ラベル用のデータ
    real_data = gray0to1(original_data)#標準化,ノイズの導入
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

#データを0-1の値にする
def gray0to1(data):
    ans_data = data /255.0
    ans_data = normalization(ans_data,1)
    return ans_data


if __name__ == '__main__':

    #データセット
    real_train_data,train_label = load_data(path_train_data,path_train_label)
    real_test_data,test_label = load_data(path_test_data,path_test_label)


    counter_result = 0
    counter_all = 0

    length = 20000
    #length = len(real_train_data)
    index_data = range(length)
    random.shuffle(index_data)
    real_train_data_copy = np.zeros((length,img_size))
    train_label_copy = np.zeros(length)

    #教師データをランダムに並び替える
    for num in range(length):

        a = index_data[num]
        real_train_data_copy[num,:] = real_train_data[a,:]
        train_label_copy[num] = train_label[a]

    #各テストデータに対する処理
    #for i in range(len(real_test_data)):
    for i in range(2500):

        counter_label = np.zeros((10))

        ####ノルムを求めてnormリストに格納####
        norm_length = np.zeros((length))
        norm_idx = np.zeros((length))
        norm_length = np.linalg.norm(real_test_data[i] - real_train_data_copy,axis = 1)
        norm_idx = train_label_copy
        idx_array = np.argsort(norm_length)

        ####どのクラスに属するか、カウントする####
        for l in range(k):
            idx = norm_idx[int(idx_array[l])]
            counter_label[int(idx)] += 1

        ####最も数の大きいインデックスを算出
        ans = np.argmax(counter_label)
        counter_all += 1

        if ans == test_label[i]:
            counter_result += 1

    #####正答率の算出####
    acc = 100.0 * counter_result / counter_all
    print acc
