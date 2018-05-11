#-*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import cityblock
import matplotlib.pyplot as plt
import copy

pic_height = 5
pic_width = 5
pixel = pic_width*pic_height
threshold = 0

#plot the data in 25*25 area
def plot(data):

    x = [[0,1,1,0],[1,2,2,1],[2,3,3,2],[3,4,4,3],[4,5,5,4]]
    y = [[0,0,1,1],[1,1,2,2],[2,2,3,3],[3,3,4,4],[4,4,5,5]]

    for j in range(len(data)):
        plt.subplot(2,3,j+1)
        plt.xlim([0,pic_width])
        plt.ylim([0,pic_height])
        for i in range(pixel):
            if(data[j][i] == 1):
                p = i/pic_width
                q = i%pic_width
                plt.fill(x[int(q)],y[(pic_height-1)-int(p)],color='blue')
    plt.show()


#change the data
def make_noise(data,ratio):
    ans = copy.deepcopy(data)
    idx = np.random.randint(0,25,int(25*ratio))
    for i in idx:
        ans[:,i] = -1*data[:,i]
    return ans


class Hopfield():

    def __init__(self,data):
        self.data = copy.deepcopy(data)
        self.weight = np.zeros((pixel,pixel))

    #learning
    def remember(self):
        for i in range(len(self.data)):
            self.weight +=  np.dot(self.data[i,:].reshape(pixel,1),self.data[i,:].reshape(1,pixel))
        self.weight /= len(self.data)
        # change to a symmetric matrix
        self.weight = (self.weight + self.weight.T)/2
        for i in range(pixel):
            self.weight[i,i] = 0

    # ipt is scalar, not vector!
    def active_func(self,val):
        if val - threshold < 0:
            return -1
        else:
            return 1

    #Lyapunov function(potential energy)
    def energy(self,data):#data is yoko
        matrix = (self.weight * np.dot(data.reshape(pixel,1),data.reshape(1,pixel)))
        sum = 0
        for i in range(pixel):
            for j in range(pixel):
                sum += matrix[i][j]
        return sum / 2 * (-1)

    #data is 1-D
    def calc(self,data):
        x = np.zeros((pixel))
        for i in range(len(x)):
            x[i] = i
        rand_array = np.random.permutation(x)
        for idx in rand_array:
            data[int(idx)] = self.active_func(np.dot(self.weight[:,int(idx)],data))
        return data

    def test(self,test_data,onoff='off'):
        answer = np.zeros((len(self.data),pixel)) #result
        dif = np.ones((len(self.data)))*25 #difference between original_data and result
        acc_num = 0# accuracy

        for i in range(len(self.data)):
            cnt = 0
            ene = 1000
            ene_pre = 10000
            while(cnt < 500 and (np.absolute(ene_pre - ene)>0)):

                ene_pre = ene
                answer[i,:] = self.calc(test_data[i])
                ene = self.energy(answer[i,:])
                dif[i] = cityblock(original_data[i,:],answer[i,:])/2
                cnt += 1
                #print(i,cnt,np.absolute(ene_pre - ene))
            if (dif[i]==0):
                acc_num += 1

        #if you plot
        if onoff=="on":
            plot(answer)

        #calculate capacaity
        dif_sum = np.sum(dif)
        return ((1 - (dif_sum /( pixel * len(dif)))),acc_num)


if __name__ == '__main__':

    repeat = 100
    '''
    original_data = np.array([[1,-1,1,-1,-1,-1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1],
        [1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1],
        [1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1],
        [1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1],
        [-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,1]])

    original_data = np.array([[1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1], #0
        [-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1],#1
        [1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1],#2
        [1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1],#3
        [1,-1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1],#4
        [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1]])#5
    '''
    original_data = np.array([[1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], #up
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],#middle
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1],#down
        [1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1],#left
        [-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1],#center
        [-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1]])#right



#######first data##############
    #plot(original_data)


#######noise to 20 ############
    #hp = Hopfield(np.array([original_data[0]]))    #one data - noise
    hp = Hopfield(original_data[0:5,:])    #six data - noise

    hp.remember()
    x_range = np.array([4,6,8,10,12,14,16,18,20])
    acc = np.zeros((len(x_range)))
    capacity = np.zeros((repeat,len(x_range)))

    #use average
    for j in range(repeat):
        for i in range(len(x_range)):
            #test_data = make_noise(np.array([original_data[0]]),x_range[i]/100)#one data
            test_data = make_noise(original_data[0:5,:],x_range[i]/100)        #six data
            capacity[j,i], acc_num = hp.test(test_data,"off")
            acc[i] += acc_num

    ave_capacity = np.average(capacity,axis=0)
    acc /= repeat*len(test_data)

    plt.subplot(1,2,1)
    plt.scatter(x_range,ave_capacity,color="blue")
    plt.xlabel("ratio of noise (%)")
    plt.ylabel("capacity")
    plt.subplot(1,2,2)
    plt.scatter(x_range,acc,color="blue")
    plt.xlabel("ratio of noise (%)")
    plt.ylabel("accuracy")

    plt.show()
    '''
#######noise to 100 ############
    #hp = Hopfield(original_data[0:2,:])    #two data - noise
    hp = Hopfield(original_data[0:4,:])    #four data - noise

    hp.remember()
    capacity = np.zeros((repeat,21))
    x_range = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
    acc = np.zeros((21))

    for j in range(repeat):
        for i in range(21):
            #test_data = make_noise(original_data[0:2],i*5/100)#two data
            test_data = make_noise(original_data[0:4],i*5/100)#two data
            capacity[j,i],acc_num = hp.test(test_data,"off")
            acc[i] += acc_num

    ave_capacity = np.average(capacity,axis=0)
    acc /= repeat*len(test_data)

    plt.subplot(1,2,1)
    plt.scatter(x_range,ave_capacity,color="blue")
    plt.xlabel("ratio of noise (%)")
    plt.ylabel("capacity")
    plt.subplot(1,2,2)
    plt.scatter(x_range,acc,color="blue")
    plt.xlabel("ratio of noise (%)")
    plt.ylabel("accuracy")
    plt.show()
    '''
