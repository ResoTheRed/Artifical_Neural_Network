import numpy as np
from genetic_alg import Data
import math
import sys

beta = [
    np.matrix([
        [0.397263,-0.440996,0.224320],
        [0.628144,0.598893,-0.628478],
        [0.595008,-0.348609,-0.208487],
        [-0.074460,0.599416,0.621480],
        [0.206494,0.347336,-0.578062]
    ]),
    np.matrix([
        [-0.214108,-0.251251,0.595624],
        [0.195364,-0.583371,-0.245872],
        [-0.121688,0.511927,-0.635016],
        [0.028181,-0.622591,0.689460]
    ]),
    np.matrix([
        [-0.4656388,  -0.4580941,   0.0034051],
        [-0.0534615,   0.3845925,  -0.2333062],
        [0.5270184,  -0.2597160,  -0.1325904],
        [0.6857013,   0.6930017,   0.4828105]
    ]),
    np.matrix([
        [0.175076,   0.317678,  -0.669430],
        [-0.599514,  -0.526939,   0.396396],
        [0.440614,   0.166052,   0.684786],
        [-0.053484,   0.644765,  -0.332944]
    ])
]

# beta = [
#     np.matrix([
#         [0.397263,-0.440996,0.224320],
#         [0.628144,0.598893,-0.628478],
#         [0.795008,-0.348609,-0.208487],
#         [-0.974460,0.599416,0.621480],
#         [0.206494,0.347336,-0.578062]
#     ]),
#     np.matrix([
#         [-0.214108,-0.251251,0.595624],
#         [0.995364,-0.583371,-0.245872],
#         [-0.121688,0.511927,-0.635016],
#         [0.028181,-0.622591,0.689460]
#     ]),
#     np.matrix([
#         [-9.4656388,  -0.4580941,   0.0034051],
#         [-9.0534615,   0.3845925,  -0.2333062],
#         [9999.5270184,  -0.2597160,  -0.1325904],
#         [0.6857013,   0.6930017,   0.4828105]
#     ]),
#     np.matrix([
#         [9999.5270184,   0.317678,  -0.669430],
#         [-0.599514,  -0.526939,   0.396396],
#         [0.440614,   0.166052,   0.684786],
#         [-0.053484,   0.644765,  -0.332944]
#     ])
# ]

class Fitness:

#  X = importdata('test_X.txt');
#  Y = importdata('test_Y.txt');
#  [Ntest, q] = size(X);
#  TestErr=0;
# %// ====== Same (or similar) code as we used before for feed-forward part (see above)
#   for j=1: Ntest		      % for loop #1		
#       Z{1} = [X(j,:) 1]';   % Load Inputs with bias=1
#       Yk   = Y(j,:)'; 	  % Load Corresponding Desired or Target output
#       % // forward propagation 
#       % //----------------------
#       for i=1:length(L)-1
#        	     T{i+1} = B{i}' * Z{i};
#              if (i+1)<length(L)
#                Z{i+1}=[(1./(1+exp(-T{i+1}))) ;1];
#              else  
#                Z{i+1}=(1./(1+exp(-T{i+1}))); 
#              end 
#       end  % //end of forward propagation 
#        Z{end}
#        TestErr= TestErr+sum(sum( ( (Yk-Z{end}). ^2), 1));   
#    end 
# TestErr= (TestErr) /(Ntest);   % //Average error of N sample after an epoch 
# mse=TestErr; 


    def __init__(self, pop, beta_len):
        self.pop_size = pop
        self.mse = self.init_mse()
        self.layer = [4,3,3,3,3]
        self.min_err = sys.maxsize
        self.data = Data(pop)
        self.Nx, self.P = np.shape(self.data.train_x)
        self.test_err = 0
        self.beta_len = beta_len

    def init_mse(self):
        mse = []
        for i in range(self.pop_size):
            mse.append((i, 100000))
        return mse

    def set_betas(self, Beta):
        self.Beta = Beta
        # self.pop_fitness('test')

    # def reset(self):
    #     self.mse = self.init_mse()
    #     self.min_err = sys.maxsize

    def pop_fitness(self, run_type):
    #     if run_type == "train":
    #         x = self.data.train_x
    #         y = self.data.train_y
    #     else:
        x = self.data.test_x
        y = self.data.test_y
    #     for i in range(self.pop_size):
        self.find_fitness(x,y)
    #     self.sort_mse()
    #     self.sort_betas()
    #     self.min_err = self.mse[0][1]
    #     return self.Beta, self.mse

    def find_fitness(self,data_x, data_y):
        for i in range(len(data_x)):
            Z = np.array(data_x[i])
            Z = np.asarray(Z, dtype='float64')
            Z = np.transpose(np.matrix(np.append(Z,[1])))
            Yk = np.asarray(data_y[i], dtype='float64')

            for j in range(self.beta_len):
                B = np.matrix(np.asarray(self.Beta[j], dtype='float64'))
                T = np.transpose(B) @ Z
                # T = self.sigmoid(T)
                if j+1 < self.beta_len:
                    T = self.sigmoid(T)
                    Z = np.append(T,[[1.0]], axis=0)
                else:
                    Z = self.sigmoid(T)
            self.test_err = self.test_err + self.get_error(Yk, Z)
        self.test_err = self.test_err/self.layer[-1]
        self.test_err = float("%.6f"% (self.test_err / len(data_x)))
        self.mse = (1, self.test_err)

    def sigmoid(self, T):
        for i in range(len(T)):
            T[i] =(1+math.exp(-float("%.6f"%T[i])))
        T = np.divide(1,T)
        return T

    def get_error(self, y, z):
        z = np.array(z)
        a = []
        for i in range(len(y)):
            a.append(y[i] - z[i])
        a = np.power(a,2)
        mse = np.sum(np.sum(a,0))
        return mse

    def set_values(self, fold,run_type):
        self.data.KFCV(fold)
        self.test_err = 0
        if run_type == 'train':
            self.Nx, self.P = np.shape(self.data.train_x)
            self.mse = sys.maxsize
        else:
            self.Nx, self.P = np.shape(self.data.test_x)
            
    def sort_mse(self):
        count = 1
        while count > 0:
            count = 0
            for i in range(len(self.mse)-1):
                if self.mse[i][1] > self.mse[i+1][1]:
                    temp = self.mse[i]
                    self.mse[i] = self.mse[i+1]
                    self.mse[i+1] = temp
                    count +=1

    def sort_betas(self):
        temp = []
        for i in range(len(self.mse)):
            temp.append(self.Beta[self.mse[i][0]])
        self.Beta = temp

fit = Fitness(1,len(beta))
fit.set_betas(beta)
fit.pop_fitness('test')
print(fit.mse)
# import time
# from threading import Thread

# def sleeper(i):
#     print( "thread %d sleeps for 5 seconds" % i)
#     time.sleep(5)
#     print( "thread %d woke up" % i)

# for i in range(10):
#     t = Thread(target=sleeper, args=(i,))
#     t.start()


