#!/usr/bin/env python
# -*- coding: utf-8 -*-

#s宣言部
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys

param_g = [0.95, 0.66] #ステップ幅用パラメータ 0-1
param_e = 0.5 #近似解精度用パラメータ 0-1
param_t = 1.0e-5 #tのステップを調整するパラメータ
num_food = 21#問3における食物種類
num_food2 = 22#問5における食物種類
path = "kondate.csv"

#データを読み込む
def load_data(path,max_or_min,num_food,cholesterol):

    data=np.loadtxt(path ,delimiter=',',dtype='float')
    A = data[0:num_food,0:11]#コレステロールは除く
    b = data[0:num_food,12]#価格データ
    c = data[22,0:11]#男性の必要摂データ

    if cholesterol==1 :
        chol = -1*data[0:num_food,11]#コレステロール
        A = np.c_[A,chol]
        c = np.r_[c,-200]

    #最小化問題のとき
    if max_or_min == "min":
        A_add = np.eye(num_food)
        A = np.c_[A,A_add] * (-1)
        b = b*(-1)
        c_add = np.zeros(num_food)
        c = np.r_[c,c_add] *(-1)

    b =b[:,np.newaxis]
    c =c[:,np.newaxis]

    return A,b,c

#計算部分
def calc(A,b,c,y0,param_g):

    cnt_array = []
    objf_array = []

    y = y0
    cnt = 0
    S_inv =  np.zeros((len(c),len(c)))
    val = -50

    while(val<0):
        cnt += 1

        #行列Sを作る
        for i in range(len(c)):
            S_inv[i,i] = 1.0 / (c[i] - np.dot(A[:,i].T,y))
        S2_inv = S_inv * S_inv

        #コレスキー分解によるdyの算出
        LL_T = np.dot(A,np.dot(S2_inv,A.T))
        L = np.linalg.cholesky(LL_T)
        u = np.linalg.solve(L,b)
        dy = np.linalg.solve(L.T,u)
        x = np.dot(S2_inv, np.dot(A.T, dy))

        #t_starをできるだけ近づける
        t_star = 0
        while(np.all((c-np.dot(A.T,(y+(t_star+param_t)*dy))) >= 0)):
            t_star += param_t

        y_plus = y + param_g*t_star*dy
        val = param_e - np.dot(b.T,(y_plus - y))

        '''
        #値の確認
        print ("cnt:{}".format(cnt))
        print ("val:{}".format(val))
        print ("t_star:{}".format(t_star))
        print ("x:{}".format(x))
        print ("y_plus:{}".format(y_plus))
        print ("bTy:{}".format(np.dot(b.T,y_plus))*(-1))
        '''
        #グラフ作成のための配列づくり
        cnt_array.append(cnt)
        objf_array.append(np.dot(b.T,y_plus)*(-1))

        #更新など
        cTx = np.dot(c.T,x)
        y = y_plus
        y_ans = np.dot(b.T,y)

    ATy =np.dot(A.T,y) * (-1)
    print ATy[11] #量

    return y,cnt_array,objf_array,cTx,y_ans

#問３用
def q3(param_g):
    A,b,c = load_data(path,"min",num_food,0)
    y0 = np.ones([num_food]) * 10
    y0 =y0[:,np.newaxis]
    y_ans,cnt_array,objf_array,cTx,y_ans = calc(A,b,c,y0,param_g)
    log_array = np.log10(objf_array - cTx)
    return cnt_array,log_array,y_ans,cTx

#問4用
def q4(param_g):
    A,b,c = load_data(path,"min",num_food,1)
    y0 = np.array([1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 1, 0.1, 10, 0.1, 0.1, 0.1, 1, 1, 1, 1, 10, 0.1, 10])
    y0 =y0[:,np.newaxis]
    y_ans,cnt_array,objf_array,cTx,y_ans = calc(A,b,c,y0,param_g)
    log_array = np.log10(objf_array - cTx)
    return cnt_array,log_array,y_ans,cTx

#問5用
def q5(param_g):
    A,b,c = load_data(path,"min",num_food2,0)
    y0 = np.ones([num_food2]) * 10
    y0 =y0[:,np.newaxis]
    y_ans,cnt_array,objf_array,cTx,y_ans = calc(A,b,c,y0,param_g)
    log_array = np.log10(objf_array - cTx)
    return cnt_array,log_array,y_ans,cTx


if __name__ == '__main__':

    #x,yの値
    cnt_array1,log_array1,y_ans1,cTx1  = q4(param_g[0])
    log_array1 = log_array1.reshape(-1,)
    cnt_array2,log_array2,y_ans2,cTx2 = q4(param_g[1])
    log_array2 = log_array2.reshape(-1,)
    print ("bTy:{} cTx:{}".format(y_ans1*(-1),cTx1*(-1)))
    print ("bTy:{} cTx:{}".format(y_ans2*(-1),cTx2*(-1)))

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))

    #0.95のときのグラフ
    ax1.plot(cnt_array1 ,log_array1 ,marker="o")
    ax1.set_title('affine scaling 0.95')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('log difference')
    ax1.grid(True)

    #0.66のときのグラフ
    ax2.plot(cnt_array2 ,log_array2 ,marker="o")
    ax2.set_title('affine scaling 0.66')
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('log difference')
    ax2.grid(True)

    fig.show()
    filename = "q4.png"
    plt.savefig(filename)
