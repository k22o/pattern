#-*- cording :utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import copy
import csv
import time

def kernel(x1,x2,type):
    #x is vertical
    if type == 0:
        #linear kernel
        ans =  np.dot(x1.T,x2)
    #RBFkernel
    elif type == 1:
        gamma = 5.0
        ans =  np.exp(-1*gamma*np.power(np.linalg.norm(x1-x2),2))
    #sigmoid kernel
    elif type == 2:
        beta = 0.06
        ans = 1/(np.exp(-beta*np.dot(x1.T,x2)))
    #polynomial kernel 3
    elif type ==3:
        ans =  np.power((np.dot(x1.T,x2)+1000),3)
    return ans

def kernel_calc(xv,Xm,type):
    #xv is vertical vector
    #Xm is matrix(datanum*features)
    ans = np.zeros([len(Xm),1])
    for i in range(len(Xm)):
        ans[i,0] = kernel(xv.T,Xm[i,:].T,type)
    return ans


class SVM():
    def __init__(self,data,label,type,upper_bound):
        self.data   = copy.deepcopy(data)           #dataset
        self.a      = np.ones([len(self.data),1])/5  #varialbes when L-dual
        self.t      = np.ones([len(self.data),1])  #teaching label
        self.be     = 1.0                           #param
        self.weight = np.zeros([np.size(self.data,1),1])
        self.bias   = 0
        self.type   =copy.deepcopy(type)
        self.upper_bound   =copy.deepcopy(upper_bound)

        for i in range(len(self.data)):
            if label[i] == "versicolor":
                self.t[i,0]=-1

    def calc(self):
        cnt = 0
        dif = 100
        pre_a =copy.deepcopy(self.a) + 100;

        #repeat prescribed count or change nothing
        while(cnt<300 and np.linalg.norm(pre_a-self.a)>0.0001):
            for i in range(len(self.a)):

                #when a[i]==0, data[i] is not support-vector
                if  self.a[i,0] !=0:
                    pre_a = copy.deepcopy(self.a)
                    dif = 1- self.t[i,0]*sum(self.a*self.t*kernel_calc(self.data[i,:].T,self.data,self.type),1) \
                        - self.be*self.t[i,0]*np.dot(self.a.T,self.t)
                    self.a[i,0] += dif*0.001
                    #1000 is upper-bound
                    #when quite a small number, regarded as 0 (counterplan of overflow)
                    if self.a[i,0] < 0.00001:
                        self.a[i,0]=0
                    if self.a[i,0] >self.upper_bound:
                        self.a[i,0]=self.upper_bound
            for i in range(len(self.a)):
                self.be += 0.00001 * np.dot(self.a.T,self.t) / 2
            cnt +=1

        self.weight  = self.a*self.t
        for i in range(len(self.data)):
            self.bias += self.t[i,0]-np.sum(self.a*self.t*kernel_calc(self.data[i,:].T,self.data,self.type))
        self.bias /= len(self.data)

    #calculate accuracy
    def test_svm(self,test_data,test_label):
        acc = 0
        for i in range(len(test_data)):
            y = np.dot(self.weight.T,kernel_calc(test_data[i,:].T,self.data,self.type))+self.bias
            if(y>0 and test_label[i]=='virginica'):
                acc +=1;
            elif (y<0 and test_label[i]=='versicolor'):
                acc +=1;
        return 100*acc/len(test_data)


######
if __name__=='__main__':
    data_moto  = np.loadtxt("iris_dataset.csv",delimiter=",",dtype="double")
    label_r =[0]*(len(data_moto))
    label_moto = ['a']*len(data_moto)
    with open("iris_name.csv",newline='') as f:#分類名が記されたデータ
        reader = csv.reader(f)
        label_r =[e for e in reader]
    for i in range(len(label_moto)):
        label_moto[i] = label_r[i][0]

    #make data and label
    for i in range (len(label_moto)):
        if label_moto[i] == 'setosa':
            idx_setosa = i
        elif label_moto[i] == 'virginica':
            idx_virginica = i
            break;
    data  = data_moto[idx_setosa+1:,:]
    label = label_moto[idx_setosa+1:]

    #difference of param
    learn_time  = [0]*7
    acc         = [0]*7
    x           = [1,2,3,4,5,6,7]
    tag = ["0.001","0.01","0.1","1","10","100","1000"]
    for i in range(len(tag)):
        start   = time.time()
        svm     = SVM(data,label,0,pow(10,x[i]-4))
        svm.calc()
        learn_time[i] = time.time()-start
        acc[i]       = svm.test_svm(data,label)

    '''
    #diffrence of kernel
    learn_time  = [0]*4
    acc         = [0]*4
    x           = [1, 3, 5, 7]
    tag =['linear','RBF','sigmoid','polynomial']
    for i in range(4):
        start   = time.time()
        svm     = SVM(data,label,i,1000)
        svm.calc()
        learn_time[i] = time.time()-start
        acc[i]       = svm.test_svm(data,label)
    '''

    print("accuracy{},time{}".format(acc,learn_time))

    plt.subplot(1,2,1)
    plt.bar(x,acc,tick_label=tag,align="center")
    plt.title("accurcay",fontsize=24)
    plt.ylim(20,100)
    plt.tick_params(labelsize=20)
    plt.subplot(1,2,2)
    plt.bar(x,learn_time,tick_label=tag,align="center")
    plt.title("time",fontsize=24)
    plt.tick_params(labelsize=20)
    plt.show()
